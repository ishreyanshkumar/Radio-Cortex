"""
Radio-Cortex O-RAN ns-3 Environment
A Gym-compatible environment for RL-based RAN congestion control
integrating with ns-O-RAN simulation platform.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import subprocess
import signal
import re
try:
    import orjson as json
except ImportError:
    import json
import socket
import time
import sys
import os
import random
import csv
from typing import Dict, List, Tuple, Optional, Union
from collections import deque
from dataclasses import dataclass
from enum import Enum
import sqlite3


class SchedulerType(Enum):
    """MAC Scheduler algorithms"""
    PROPORTIONAL_FAIR = 0
    ROUND_ROBIN = 1
    MAX_THROUGHPUT = 2


def _to_python(obj):
    """Recursively convert numpy types and non-JSON values (NaN/Inf) to plain Python."""
    if isinstance(obj, np.ndarray):
        return [_to_python(v) for v in obj.tolist()]
    if isinstance(obj, (np.generic, float)):
        val = obj.item() if hasattr(obj, 'item') else obj
        if np.isnan(val) or np.isinf(val): return 0.0
        return val
    if isinstance(obj, dict):
        return {str(k): _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    return obj



class SimulationDB:
    """
    Unified SQLite logger — stores step-wise simulation traces and run metadata.
    Inlined into oran_ns3_env for zero extra imports.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)')
        c.execute('''
            CREATE TABLE IF NOT EXISTS steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode INTEGER, step INTEGER, timestamp REAL,
                state TEXT, action TEXT, reward REAL,
                metrics TEXT, next_state TEXT
            )''')
        conn.commit()
        conn.close()

    def log_metadata(self, key: str, value):
        conn = sqlite3.connect(self.db_path)
        # Always coerce to string — avoids SQLite BLOB storage of ints/bytes
        if isinstance(value, bytes):
            val = value.decode('utf-8', errors='replace')
        elif isinstance(value, str):
            val = value
        else:
            val = str(value)  # int, float, etc. → plain string
        conn.execute('INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)', (key, val))
        conn.commit(); conn.close()

    def log_step(self, step_data: dict):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            'INSERT INTO steps (episode, step, timestamp, state, action, reward, metrics, next_state) VALUES (?,?,?,?,?,?,?,?)',
            (
                step_data.get('episode', 0),
                step_data.get('step'),
                step_data.get('timestamp', time.time()),
                json.dumps(step_data.get('state', [])),
                json.dumps(step_data.get('action', [])),
                step_data.get('reward', 0.0),
                json.dumps(step_data.get('metrics', {})),
                json.dumps(step_data.get('next_state', [])),
            )
        )
        conn.commit(); conn.close()

    def log_evaluation(self, results: dict):
        for k, v in results.items():
            self.log_metadata(f'eval_{k}', v)

@dataclass
class NS3Config:
    """ns-3 simulation configuration"""
    num_ues: int = 20
    num_cells: int = 3
    sim_time: float = 10.0  # seconds
    seed: int = 42

    e2_port: int = 36421
    kpm_interval_ms: int = 100  # E2SM-KPM reporting interval
    system_bandwidth_mhz: float = 10.0 # System Bandwidth
    # Scenario Selection (Multi-scenario Training)
    scenarios: Optional[Union[List[str], Dict[str, float]]] = None  # List of names OR {name: weight}
    scenario: str = 'flash_crowd'     # Currently active scenarion
    # Topic suffix for parallel environment isolation (e.g., "_0", "_1")
    topic_suffix: str = ""
    # Shared run timestamp — set once at training start so all envs write to the same CSV.
    # Leave empty to auto-generate per-env (legacy behaviour).
    run_timestamp: str = ""
    # Verbosity control for CLI output
    verbose: bool = True
    # Testing: Randomly shuffle UE indices in the state (Order Scramble)
    shuffle_ues: bool = False

    def __post_init__(self):
        """Automatically adjust configuration defaults based on selected scenarios."""
        # 1. Sync scenario with scenarios list if necessary
        if self.scenarios and self.scenario not in (self.scenarios if isinstance(self.scenarios, list) else self.scenarios.keys()):
            # If current scenario isn't in the list, default to the first one available
            self.scenario = list(self.scenarios)[0] if isinstance(self.scenarios, (list, dict)) else self.scenario

        # 2. Major Bug Fix: Align num_ues/num_cells with C++ overrides for scenario requirements
        # Some ns-3 scenarios forcibly require specific counts to function (e.g., hexagonal layouts).
        target_ue_count = self.num_ues
        target_cell_count = self.num_cells
        
        # Check all possible scenarios in multi-scenario training
        check_scenarios = [self.scenario]
        if self.scenarios:
            check_scenarios.extend(list(self.scenarios) if isinstance(self.scenarios, list) else list(self.scenarios.keys()))
            
        # IoT Tsunami overrides (100 UEs)
        if 'iot_tsunami' in check_scenarios:
            target_ue_count = max(target_ue_count, 100)
        
        # Mobility/Topo scenarios (require 7 cells for the hexagonal cluster)
        topo_scenarios = ['mobility_storm', 'urban_canyon', 'ping_pong', 'commuter_rush']
        if any(s in check_scenarios for s in topo_scenarios):
            target_cell_count = max(target_cell_count, 7)
            
        if target_ue_count != self.num_ues:
            if self.verbose:
                print(f"⚠️  [NS3Config] Scenario Scale detected. Forcing num_ues={target_ue_count} to align with C++.")
            self.num_ues = target_ue_count
            
        if target_cell_count != self.num_cells:
            if self.verbose:
                print(f"⚠️  [NS3Config] Topology Scenario detected. Forcing num_cells={target_cell_count} to prevent crash (hexagonal layout requirement).")
            self.num_cells = target_cell_count


@dataclass
class E2Message:
    """E2 interface message format"""
    timestamp: float
    ue_metrics: Dict[int, Dict]  # {ue_id: {throughput, delay, loss, sinr}}
    cell_metrics: Dict[int, Dict]  # {cell_id: {queue_len, rb_util, power}}


class RewardEngine:
    """
    Stationary Reward Engine for Radio-Cortex O-RAN.
    """

    def __init__(self, config: NS3Config):
        self.config = config

        # ── Weights ──────────────────────────────────────────────
        self.W_TPUT      = 8.0      # Reduced to prioritize QoS
        self.W_DELAY_LIN = 4.0      # Increased to penalize latency harder
        self.W_LOSS      = 8.0      # Increased to kill packet loss
        self.W_LOAD      = 4.0      # Load Balancing (Boosted for aggressive CIO)
        self.W_ENERGY    = 0.1      # Energy Efficiency (Tx Power Regularization)
        self.W_CIO       = 0.4      # CIO Regularization (Center at 0dB)
        self.BIAS        = 1.0      # Positive bias for survival

        # ── Thresholds / Normalizers ─────────────────────────────
        self.T_MAX     = 10.0      # Throughput normalizer (Mbps)
        self.D_MAX     = 65.0      # Delay normalizer (ms)
        self.EPSILON   = 1e-6      # Safe log
        self.MIN_TPUT_SUCCESS = 1.0 # 1 Mbps = "satisfied" UE

        # ── Clip bounds (Stage 1 — per component) ───────────────
        self.CLIP_TPUT   = (-0.5, 50.0)
        self.CLIP_DELAY  = (-50.0, 0.0)
        self.CLIP_LOAD   = (-2.0, 0.0)

        # ── Clip bounds (Stage 2 — total) ────────────────────────
        self.CLIP_TOTAL = (-100.0, 50.0)

    def compute(self, e2_msg: E2Message, action=None, prev_action=None, action_space=None, action_dict: Dict = None) -> Tuple[float, Dict[str, float]]:
        empty = self._empty_breakdown()
        if not e2_msg.ue_metrics:
            return 0.0, empty

        # 1. UE UTILITY (Core Signals)
        tputs  = np.array([m['throughput']  for m in e2_msg.ue_metrics.values()])
        delays = np.array([m['delay']       for m in e2_msg.ue_metrics.values()])
        losses = np.array([m['packet_loss'] for m in e2_msg.ue_metrics.values()])

        # Throughput (Log Utility for Fairness)
        r_tput = float(np.mean(np.log(1.0 + tputs / self.T_MAX + self.EPSILON))) * self.W_TPUT
        r_tput = float(np.clip(r_tput, *self.CLIP_TPUT))

        # Delay (Strictly Linear to prevent gradient explosions)
        d_norm = np.minimum(delays / self.D_MAX, 1.0)
        r_delay = float(-np.mean(self.W_DELAY_LIN * d_norm))
        r_delay = float(np.clip(r_delay, *self.CLIP_DELAY))

        # Packet Loss (Strictly Linear)
        mean_loss = float(np.mean(losses))
        loss_penalty_raw = mean_loss * 4.0 
        r_loss = float(np.clip(-loss_penalty_raw * self.W_LOSS, -25.0, 0.0))

        # 2. NETWORK UTILITY (Load Balancing via CIO)
        r_load = 0.0
        if e2_msg.cell_metrics:
            loads = np.array([m['cell_load'] for m in e2_msg.cell_metrics.values()])
            r_load = float(-np.std(loads)) * self.W_LOAD
            r_load = float(np.clip(r_load, *self.CLIP_LOAD))

            # Energy Efficiency (Tx Power Regularization)
            tx_powers = np.array([m['tx_power'] for m in e2_msg.cell_metrics.values()])
            norm_power = (tx_powers - 10.0) / 36.0
            r_energy = float(-np.mean(norm_power)) * self.W_ENERGY
            r_energy = float(np.clip(r_energy, -1.0, 0.0))
        else:
            r_energy = 0.0

        # SLA Bonus (Pass/Fail Cliff)
        satisfied_mask = (tputs >= 1.0) & (delays <= 100.0)
        r_sla = float(np.sum(satisfied_mask)) * 0.5

        # CIO Regularization (Center at 0dB)
        r_cio = 0.0
        if action_dict:
             cios = [c['cell_individual_offset_db'] for c in action_dict['cell']]
             norm_cios = np.abs(np.array(cios)) / 6.0
             r_cio = float(-np.mean(norm_cios)) * self.W_CIO

        # 3. AGGREGATE
        total = float(np.clip(r_tput + r_delay + r_loss + r_load + r_energy + r_sla + r_cio + self.BIAS, *self.CLIP_TOTAL))

        # 4. DIAGNOSTICS
        jains = float(self._jains_index(tputs))
        success_rate = np.sum(tputs > self.MIN_TPUT_SUCCESS) / max(len(tputs), 1)

        breakdown = {
            'r_total': total, 'r_tput': r_tput, 'r_delay': r_delay, 'r_loss': r_loss,
            'r_load': r_load, 'r_energy': r_energy, 'r_sla': r_sla, 'r_cio': r_cio,
            'r_queue': 0.0, 'r_smooth': 0.0,
            'jains': jains, 'p95_delay': float(np.percentile(delays, 95)) if len(delays)>0 else 0.0,
            'avg_throughput': float(np.mean(tputs)), 'avg_delay': float(np.mean(delays)),
            'avg_loss': mean_loss, 'z_success': success_rate,
        }
        return total, breakdown

    @staticmethod
    def _jains_index(x: np.ndarray) -> float:
        s = float(np.sum(x))
        if s <= 0:
            return 0.0
        return (s ** 2) / (len(x) * float(np.sum(x ** 2)))

    def _empty_breakdown(self) -> Dict[str, float]:
        return {
            'r_total': 0.0, 'r_tput': 0.0, 'r_delay': 0.0, 'r_loss': 0.0,
            'r_energy': 0.0, 'r_load': 0.0, 'r_queue': 0.0,
            'r_smooth': 0.0, 
            'jains': 0.0, 'p95_delay': 0.0,
            'avg_throughput': 0.0, 'avg_delay': 0.0, 'avg_loss': 0.0,
            'z_success': 0.0, 'r_sla': 0.0, 'r_cio': 0.0,
        }


class NS3Interface:
    def __init__(self, config: NS3Config):
        self.config = config
        self.ns3_process = None
        self.kafka_consumer = None
        self.kafka_producer = None
        self.current_step = 0
        self.last_kpm_ts = None
        
    def start_simulation(self):
        # Resolve ns3 script path
        ns3_path = 'ns3'
        import glob
        if not os.path.exists(ns3_path):
            candidates = sorted(glob.glob('ns-3-allinone/ns-3.*/ns3'), reverse=True)
            if candidates: ns3_path = candidates[0]
        
        ns3_path = os.path.abspath(ns3_path)
        ns3_dir = os.path.dirname(ns3_path)
        
        binary_path = None
        binary_name_opt_pattern = "ns3.*-oran-congestion-scenario-optimized"
        binary_name_default_pattern = "ns3.*-oran-congestion-scenario-default"
        
        candidate_patterns = [
            os.path.join(ns3_dir, "build/scratch", binary_name_opt_pattern),
            os.path.join(ns3_dir, "build/optimized/scratch", binary_name_opt_pattern),
            os.path.join(ns3_dir, "build/scratch", binary_name_default_pattern),
        ]
        
        for pattern in candidate_patterns:
            matches = glob.glob(pattern)
            if matches:
                binary_path = matches[0]
                break
        
        if binary_path:
            ns3_cmd = [
                os.path.abspath(binary_path),
                f'--numUes={self.config.num_ues}',
                f'--numCells={self.config.num_cells}',
                f'--simTime={self.config.sim_time}',
                f'--seed={self.config.seed}',
                f'--kpmInterval={self.config.kpm_interval_ms}',
                f'--enableE2=true',
                f'--scenario={self.config.scenario}',
                f'--bandwidthRbs={int(self.config.system_bandwidth_mhz * 5)}',
            ]
            if self.config.topic_suffix:
                ns3_cmd.append(f'--topicSuffix={self.config.topic_suffix}')
        else:
            ns3_cmd = [
                sys.executable, os.path.basename(ns3_path), 'run',
                f'scratch/oran-congestion-scenario',
                '--',
                f'--numUes={self.config.num_ues}',
                f'--numCells={self.config.num_cells}',
                f'--simTime={self.config.sim_time}',
                f'--seed={self.config.seed}',
                f'--kpmInterval={self.config.kpm_interval_ms}',
                '--enableE2=true',
                f'--scenario={self.config.scenario}',
                f'--bandwidthRbs={int(self.config.system_bandwidth_mhz * 5)}',
                f'--topicSuffix={self.config.topic_suffix}'
            ]
        
        os.makedirs("logs", exist_ok=True)
        self.log_file_out = open(os.path.join("logs", f"ns3_out{self.config.topic_suffix}.log"), "w")
        self.log_file_err = open(os.path.join("logs", f"ns3_err{self.config.topic_suffix}.log"), "w")
        
        env = os.environ.copy()
        lib_paths = [os.path.join(ns3_dir, "build/lib"), os.path.join(ns3_dir, "build")]
        valid_lib_paths = [p for p in lib_paths if os.path.exists(p)]
        if valid_lib_paths:
            env['LD_LIBRARY_PATH'] = ':'.join(valid_lib_paths) + ':' + env.get('LD_LIBRARY_PATH', '')

        if not self.kafka_consumer:
            self._connect_kafka()
        else:
            # Drain any stale messages from the previous run
            try:
                while True:
                    records = self.kafka_consumer.poll(timeout_ms=100)
                    if not records:
                        break
            except Exception:
                pass
        
        self.ns3_process = subprocess.Popen(
            ns3_cmd, cwd=ns3_dir, stdout=self.log_file_out, stderr=self.log_file_err,
            preexec_fn=os.setsid, env=env
        )
        self.last_kpm_ts = None
        
    def _connect_kafka(self):
        from kafka import KafkaConsumer, KafkaProducer
        kpm_topic = f'e2_kpm_stream{self.config.topic_suffix}'
        rc_topic = f'e2_rc_control{self.config.topic_suffix}'
        self.kafka_consumer = KafkaConsumer(
            kpm_topic, bootstrap_servers=[os.getenv('KAFKA_BOOTSTRAP', 'localhost:9092')],
            auto_offset_reset='latest', enable_auto_commit=False,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            group_id=f'oran_rl_agent{self.config.topic_suffix}_{int(time.time())}'
        )
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=[os.getenv('KAFKA_BOOTSTRAP', 'localhost:9092')],
            value_serializer=lambda x: (d if isinstance(d := json.dumps(x), bytes) else d.encode('utf-8'))
        )
        self._rc_topic = rc_topic
    
    def receive_kpm_report(self, wait_for_new: bool = True, max_wait_s: float = 10.0) -> E2Message:
        deadline = time.time() + max_wait_s
        while time.time() < deadline:
            records = self.kafka_consumer.poll(timeout_ms=10)
            if records:
                for partition, messages in records.items():
                    if messages:
                        candidate = messages[-1]
                        if self.last_kpm_ts is None or candidate.timestamp > self.last_kpm_ts:
                            self.last_kpm_ts = candidate.timestamp
                            return self._parse_kpm(candidate.value)
        if self.ns3_process and self.ns3_process.poll() is not None:
             raise RuntimeError("ns-3 died")
        raise TimeoutError("Kafka KPM timeout")
            
    def _parse_kpm(self, kpm_data):
        ue_metrics = {}
        for ue_id in range(self.config.num_ues):
            ue_metrics[ue_id] = {
                'throughput': kpm_data.get(f'ue_{ue_id}_tput', 0.0),
                'delay': kpm_data.get(f'ue_{ue_id}_delay', 0.0),
                'packet_loss': kpm_data.get(f'ue_{ue_id}_loss', 0.0),
                'sinr': kpm_data.get(f'ue_{ue_id}_sinr', 0.0),
                'rsrp': kpm_data.get(f'ue_{ue_id}_rsrp', -140.0),
                'serving_cell': kpm_data.get(f'ue_{ue_id}_cell', -1),
                'handover_successes': kpm_data.get(f'ue_{ue_id}_ho_succ', 0),
            }
        cell_metrics = {}
        for cell_id in range(self.config.num_cells):
            cell_metrics[cell_id] = {
                'queue_length': kpm_data.get(f'cell_{cell_id}_queue', 0),
                'rb_utilization': kpm_data.get(f'cell_{cell_id}_rb_util', 0.0),
                'tx_power': kpm_data.get(f'cell_{cell_id}_power', 46.0),
                'num_connected_ues': kpm_data.get(f'cell_{cell_id}_ues', 0),
                'cell_load': kpm_data.get(f'cell_{cell_id}_load', 0.0),
                'avg_rb_request': kpm_data.get(f'cell_{cell_id}_avg_rb_req', 0.0),
            }
        return E2Message(timestamp=time.time(), ue_metrics=ue_metrics, cell_metrics=cell_metrics)
    
    def send_rc_control(self, actions: Dict):
        flat = {}
        for cell_dict in actions.get('cell', []):
            cid = cell_dict.get('cell_id', 0)
            flat[f'cell_{cid}'] = 1
            for k, v in cell_dict.items():
                if k != 'cell_id': flat[f'cell_{cid}_{k}'] = v
        rc_message = {'type': 'E2SM_RC', 'timestamp': time.time(), **flat}
        try:
            self.kafka_producer.send(self._rc_topic, rc_message)
            self.kafka_producer.flush(timeout=5.0)
        except Exception as e:
            print(f"Error sending RC: {e}")
    
    def stop_simulation(self, close_kafka: bool = True):
        if close_kafka:
            if self.kafka_consumer: self.kafka_consumer.close(); self.kafka_consumer = None
            if self.kafka_producer: self.kafka_producer.close(); self.kafka_producer = None
        if self.ns3_process:
            try: os.killpg(os.getpgid(self.ns3_process.pid), signal.SIGKILL)
            except: pass
            self.ns3_process.wait(timeout=1)
            self.ns3_process = None


class ORANns3Env(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, config: Optional[NS3Config] = None):
        super().__init__()
        self.config = config or NS3Config()
        self.ns3 = NS3Interface(self.config)
        self.reward_engine = RewardEngine(self.config)
        
        suffix = self.config.topic_suffix or '_0'
        env_ids = re.findall(r'\d+', suffix)
        self._env_id = int(env_ids[-1]) if env_ids else 0
        
        try:
            from datetime import datetime
            ts = self.config.run_timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
            self._sim_db_path = f'results/simulation_data_{ts}.db'
            if self._env_id == 0:
                self._sim_db = SimulationDB(self._sim_db_path)
                self._sim_db.log_metadata('scenario', self.config.scenario)
                self._sim_db.log_metadata('num_ues', self.config.num_ues)
                self._sim_db.log_metadata('num_cells', self.config.num_cells)
                self._sim_db.log_metadata('sim_time', self.config.sim_time)
            else:
                self._sim_db = None
        except:
            self._sim_db = None

        self.features_per_cell = 16
        self.n_stack = 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.config.num_cells * 16 * 3,), dtype=np.float32)
        self.actions_per_cell = 3
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.config.num_cells * 3,), dtype=np.float32)
        
        self.current_step = 0
        self._episode = 0
        self.max_steps = int(self.config.sim_time * 10)
        self.episode_metrics = []
        self._prev_obs_raw = None
        self.prev_obs = np.zeros(self.config.num_cells * 16, dtype=np.float32)  # safe default
        self._obs_history = deque([np.zeros(self.config.num_cells * 16) for _ in range(3)], maxlen=3)
        self.current_params = {'cell': {c: {'tx_power': 46.0, 'cio': 0.0, 'ttt': 192.0} for c in range(self.config.num_cells)}}

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        if seed is not None: self.config.seed = seed
        self._prev_obs_raw = None
        self._obs_history = deque([np.zeros(self.config.num_cells * 16) for _ in range(3)], maxlen=3)
        
        if self.config.scenarios:
            self.config.scenario = random.choice(list(self.config.scenarios) if isinstance(self.config.scenarios, list) else list(self.config.scenarios.keys()))
            self.ns3.config.scenario = self.config.scenario

        self.ns3.stop_simulation(close_kafka=False)
        self.ns3.start_simulation()
        
        try:
            e2_msg = self.ns3.receive_kpm_report(max_wait_s=20.0, wait_for_new=True)
            state = self._extract_state(e2_msg)
            if self._sim_db:
                # Log step 0 with dummy metrics to satisfy visualizer
                dummy_metrics = self.reward_engine._empty_breakdown()
                dummy_metrics['e2_data'] = self._serialize_e2_message(e2_msg)
                self._sim_db.log_step({
                    'episode': self._episode, 'step': 0, 'state': _to_python(state),
                    'timestamp': time.time(), 'metrics': _to_python(dummy_metrics)
                })
        except Exception as e:
            print(f"Reset failed: {e}")
            state = np.zeros(self.observation_space.shape[0]//3)

        self._obs_history.append(state)
        stacked_state = np.concatenate(list(self._obs_history), axis=0)
        self.prev_obs = state
        self.current_step = 0
        self._episode += 1
        return stacked_state, {'scenario': self.config.scenario}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        try:
            rc_actions = self._parse_action(action)
            self.ns3.send_rc_control(rc_actions)
            
            e2_msg = self.ns3.receive_kpm_report(wait_for_new=True, max_wait_s=60.0)
            next_state = self._extract_state(e2_msg)
            
            self._obs_history.append(next_state)
            stacked_state = np.concatenate(list(self._obs_history), axis=0)
            
            reward, breakdown = self._compute_reward(e2_msg, rc_actions)
            
            self.current_step += 1
            terminated = self.current_step >= self.max_steps
            if self.ns3.ns3_process and self.ns3.ns3_process.poll() is not None: terminated = True
            
            info = {
                'step': self.current_step,
                'e2_metrics': self._serialize_e2_message(e2_msg),
                'actions_applied': rc_actions,
                **breakdown
            }
            
            if self._sim_db:
                try:
                    enriched = _to_python(breakdown)
                    enriched['e2_data'] = _to_python(self._serialize_e2_message(e2_msg))
                    enriched['actions_applied'] = _to_python(rc_actions)
                    self._sim_db.log_step({
                        'episode': self._episode, 'step': self.current_step,
                        'state': _to_python(self.prev_obs),
                        'action': _to_python(action),
                        'reward': round(float(np.mean(reward)), 4),
                        'next_state': _to_python(next_state),
                        'metrics': enriched, 'timestamp': time.time()
                    })
                except Exception as db_err:
                    print(f"[SimDB] step log failed: {db_err}", flush=True)

            self.prev_obs = next_state
            return stacked_state, reward, terminated, False, info
        except Exception as e:
            print(f"Step error: {e}")
            return np.zeros(self.observation_space.shape), 0.0, True, False, {"error": str(e)}

    @staticmethod
    def _serialize_e2_message(e2_msg: E2Message) -> dict:
        if e2_msg is None: return {}
        return {'timestamp': e2_msg.timestamp, 'ue_metrics': e2_msg.ue_metrics, 'cell_metrics': e2_msg.cell_metrics}

    def _extract_state(self, e2_msg: E2Message) -> np.ndarray:
        cell_stats = {c: {'tputs': [], 'delays': [], 'losses': []} for c in range(self.config.num_cells)}
        for m in e2_msg.ue_metrics.values():
            c_id = m.get('serving_cell', -1)
            if 0 <= c_id < self.config.num_cells:
                cell_stats[c_id]['tputs'].append(m.get('throughput', 0.0))
                cell_stats[c_id]['delays'].append(m.get('delay', 0.0))
                cell_stats[c_id]['losses'].append(m.get('packet_loss', 0.0))
        
        state = np.zeros(self.config.num_cells * 16, dtype=np.float32)
        current_obs_raw = {}
        for c in range(self.config.num_cells):
            cm = e2_msg.cell_metrics.get(c, {})
            idx = c * 16
            state[idx:idx+5] = [
                np.clip(cm.get('queue_length', 0)/1000.0, -1, 1),
                np.clip(cm.get('rb_utilization', 0.0), -1, 1),
                np.clip((cm.get('tx_power', 46.0)-10)/36.0, -1, 1),
                np.clip(cm.get('cell_load', 0.0)/20.0, -1, 1),
                np.clip(cm.get('avg_rb_request', 0.0)/100.0, -1, 1)
            ]
            s = cell_stats[c]
            if s['tputs']:
                state[idx+5:idx+11] = [
                    np.clip(np.mean(s['tputs'])/10.0, -1, 1),
                    np.clip(np.mean(s['delays'])/100.0, -1, 1),
                    np.clip(np.mean(s['losses']), -1, 1),
                    np.clip(np.max(s['delays'])/100.0, -1, 1),
                    np.clip(np.max(s['losses']), -1, 1),
                    0.8 # Placeholder jains
                ]
            state[idx+11] = len(s['tputs'])/50.0
            current_obs_raw[c] = {'q': cm.get('queue_length',0), 'rb': cm.get('rb_utilization',0.0), 'd': np.mean(s['delays']) if s['delays'] else 0}
            idx += 16
        return state

    def _parse_action(self, action: np.ndarray) -> Dict:
        action = np.clip(action.flatten(), -1, 1)
        rc_actions = {'cell': []}
        for c in range(self.config.num_cells):
            cell_act = action[c*3 : (c+1)*3]
            rc_actions['cell'].append({
                'cell_id': c,
                'tx_power_dbm': float(28.0 + cell_act[0]*18.0),
                'cell_individual_offset_db': float(cell_act[1]*6.0),
                'time_to_trigger_ms': float(640.0 * (1.1 - cell_act[2]))
            })
        return rc_actions
    
    def _compute_reward(self, e2_msg: E2Message, action_dict: Dict = None) -> Tuple[float, Dict]:
        return self.reward_engine.compute(e2_msg, action_dict=action_dict)

    def close(self):
        self.ns3.stop_simulation()


def create_oran_env(config: Optional[NS3Config] = None) -> ORANns3Env:
    """Factory function for creating O-RAN environment"""
    return ORANns3Env(config)

if __name__ == "__main__":
    env = ORANns3Env(NS3Config(sim_time=1.0))
    obs, info = env.reset()
    for _ in range(5):
        obs, reward, done, trunc, info = env.step(env.action_space.sample())
        if done: break
    env.close()
    print("Done")
