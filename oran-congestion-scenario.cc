/**
 * Radio-Cortex O-RAN Congestion Control Scenario
 *
 * This ns-3 scenario implements:
 * 1. Multi-cell LTE/NR network with configurable topology
 * 2. E2 interface via KAFKA for external RL control (E2SM-KPM + E2SM-RC)
 * 3. Congestion scenarios: flash crowds, mobility storms, traffic bursts
 * 4. Real-time KPM reporting for RL state observation
 *
 * Usage: ./ns3 run "oran-congestion-scenario --numUes=20 --numCells=3"
 */

#include "ns3/applications-module.h"
#include "ns3/config-store-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/lte-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/seq-ts-header.h"
#include "ns3/udp-server.h"
#include <ns3/lte-enb-mac.h>
#include <ns3/lte-enb-net-device.h>
#include <ns3/lte-enb-phy.h>
#include <ns3/lte-enb-rrc.h>
#include <ns3/lte-ue-mac.h>
#include <ns3/lte-ue-net-device.h>
#include <ns3/lte-ue-phy.h>
#include <ns3/lte-ue-rrc.h>
#include <ns3/pointer.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime> // For timestamp generation
#include <iostream>
#include <librdkafka/rdkafka.h>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("RadioCortexOranScenario");

template <typename Tag, typename Tag::type P> struct PrivateMember {
  friend typename Tag::type get(Tag) { return P; }
};

struct LteEnbRrcMeasConfigTag {
  typedef LteRrcSap::MeasConfig LteEnbRrc::*type;
};
template struct PrivateMember<LteEnbRrcMeasConfigTag,
                              &LteEnbRrc::m_ueMeasConfig>;
LteRrcSap::MeasConfig LteEnbRrc::*get(LteEnbRrcMeasConfigTag);

struct LteEnbRrcUeMapTag {
  typedef std::map<uint16_t, Ptr<UeManager>> LteEnbRrc::*type;
};
template struct PrivateMember<LteEnbRrcUeMapTag, &LteEnbRrc::m_ueMap>;
std::map<uint16_t, Ptr<UeManager>> LteEnbRrc::*get(LteEnbRrcUeMapTag);

struct UeManagerReconfigTag {
  typedef void (UeManager::*type)();
};
template struct PrivateMember<UeManagerReconfigTag,
                              &UeManager::ScheduleRrcConnectionReconfiguration>;
void (UeManager::*get(UeManagerReconfigTag))();

// ============================================================================
// E2 Interface Manager - Handles KPM reporting and RC commands via Kafka
// ============================================================================

// Struct to hold accumulated metrics for a UE
struct UeMetricAccumulator {
  uint64_t bytesRx{0};
  uint32_t packetsRx{0};
  double sumLatency{0.0}; // milliseconds
  double sumSinr{0.0};    // linear
  uint32_t sinrSamples{0};
  double sumRsrp{0.0}; // dBm
  uint32_t rsrpSamples{0};
  double sumRsrpSq{0.0};
  double sumRsrq{0.0}; // dB
  uint32_t rsrqSamples{0};
  double sumRsrqSq{0.0};
  uint32_t ulRbCountSum{0};
  uint32_t ulRbSamples{0};
  uint32_t packetsLost{0};
  uint64_t rbsAllocated{0}; // Estimated
  int32_t servingCellId{-1};
  uint32_t handoverAttempts{0};
  uint32_t handoverSuccesses{0};

  void Reset() {
    bytesRx = 0;
    packetsRx = 0;
    sumLatency = 0.0;
    sumSinr = 0.0;
    sinrSamples = 0;
    sumRsrp = 0.0;
    rsrpSamples = 0;
    sumRsrpSq = 0.0;
    sumRsrq = 0.0;
    rsrqSamples = 0;
    sumRsrqSq = 0.0;
    ulRbCountSum = 0;
    ulRbSamples = 0;
    packetsLost = 0;
    rbsAllocated = 0;
    servingCellId = -1;
    handoverAttempts = 0;
    handoverSuccesses = 0;
  }
};

// Singleton-like helper to collect metrics from traces
class MetricCollector : public SimpleRefCount<MetricCollector> {
public:
  MetricCollector();

  // Register traces
  void RegisterUeTraces(NodeContainer ues);
  void RegisterEnbTraces(NodeContainer enbs);

  // Callbacks
  void ReportUeSinr(uint16_t cellId, uint16_t rnti, double rsrp, double sinr,
                    uint8_t componentCarrierId);
  void ReportUeMeasurements(uint16_t rnti, uint16_t cellId, double rsrp,
                            double rsrq, bool isServingCell,
                            uint8_t componentCarrierId);
  void ReportUlPhyResourceBlocks(uint16_t rnti, const std::vector<int> &rbs);
  void ReportDlScheduling(DlSchedulingCallbackInfo info);
  void ReportAppRx(uint32_t ueIndex, Ptr<const Packet> packet,
                   const Address &from);

  // Handover Callbacks
  void ReportHandoverStart(uint64_t imsi, uint16_t cellId, uint16_t rnti,
                           uint16_t targetCellId);
  void ReportHandoverEndOk(uint64_t imsi, uint16_t cellId, uint16_t rnti);

  // Accessors
  std::map<uint32_t, UeMetricAccumulator> GetAndResetUeMetrics();
  void UpdateRntiMapping(uint32_t ueIndex, uint16_t rnti);

private:
  std::map<uint32_t, UeMetricAccumulator> m_ueMetrics; // Key: UE index
  std::map<uint32_t, Ptr<UdpServer>> m_ueUdpServers;   // Key: UE index
  std::map<uint32_t, uint32_t>
      m_lastLost; // Key: UE index, value: cumulative lost
  std::map<uint16_t, uint32_t> m_rntiToUeIndex; // Key: RNTI -> UE index
  std::mutex m_mutex;
};

// Global pointer to collector for callbacks (since callbacks are static/bound)
Ptr<MetricCollector> g_metricCollector;

MetricCollector::MetricCollector() {}

void MetricCollector::UpdateRntiMapping(uint32_t ueIndex, uint16_t rnti) {
  if (rnti == 0) {
    return;
  }

  std::lock_guard<std::mutex> lock(m_mutex);
  m_rntiToUeIndex[rnti] = ueIndex;
}

void MetricCollector::RegisterUeTraces(NodeContainer ues) {
  // Manual iteration to ensure connection
  for (uint32_t i = 0; i < ues.GetN(); ++i) {
    Ptr<Node> node = ues.Get(i);
    // Find UdpServer app
    for (uint32_t k = 0; k < node->GetNApplications(); ++k) {
      Ptr<Application> app = node->GetApplication(k);
      Ptr<UdpServer> server = DynamicCast<UdpServer>(app);
      if (server) {
        // Assume RNTI maps to node ID + 1 for now
        Ptr<LteUeNetDevice> lteDev = 0;
        for (uint32_t j = 0; j < node->GetNDevices(); ++j) {
          lteDev = node->GetDevice(j)->GetObject<LteUeNetDevice>();
          if (lteDev) {
            break;
          }
        }
        if (lteDev) {
          uint16_t rnti = lteDev->GetRrc()->GetRnti();
          if (rnti != 0) {
            UpdateRntiMapping(i, rnti);
          }

          server->TraceConnectWithoutContext(
              "Rx", MakeCallback(&MetricCollector::ReportAppRx, this, i));
          m_ueUdpServers[i] = server;
          m_lastLost[i] = 0;
          NS_LOG_INFO("Registered App Trace for UE index " << i);
        }
      }
    }

    // PHY Traces
    for (uint32_t j = 0; j < node->GetNDevices(); ++j) {
      Ptr<NetDevice> dev = node->GetDevice(j);
      Ptr<LteUeNetDevice> lteDev = dev->GetObject<LteUeNetDevice>();
      if (lteDev) {
        // PHY Trace (SINR)
        Ptr<LteUePhy> phy = lteDev->GetPhy();
        if (phy) {
          phy->TraceConnectWithoutContext(
              "ReportCurrentCellRsrpSinr",
              MakeCallback(&MetricCollector::ReportUeSinr, this));
          phy->TraceConnectWithoutContext(
              "ReportUeMeasurements",
              MakeCallback(&MetricCollector::ReportUeMeasurements, this));
          phy->TraceConnectWithoutContext(
              "ReportUlPhyResourceBlocks",
              MakeCallback(&MetricCollector::ReportUlPhyResourceBlocks, this));
        }

        // RRC Traces for Handover
        Ptr<LteUeRrc> rrc = lteDev->GetRrc();
        if (rrc) {
          rrc->TraceConnectWithoutContext(
              "HandoverStart",
              MakeCallback(&MetricCollector::ReportHandoverStart, this));
          rrc->TraceConnectWithoutContext(
              "HandoverEndOk",
              MakeCallback(&MetricCollector::ReportHandoverEndOk, this));
        }
      }
    }
  }
}

void MetricCollector::RegisterEnbTraces(NodeContainer enbs) {
  for (uint32_t i = 0; i < enbs.GetN(); ++i) {
    Ptr<Node> enb = enbs.Get(i);
    for (uint32_t j = 0; j < enb->GetNDevices(); ++j) {
      Ptr<LteEnbNetDevice> lteDev =
          enb->GetDevice(j)->GetObject<LteEnbNetDevice>();
      if (lteDev) {
        Ptr<LteEnbMac> mac = lteDev->GetMac();
        if (mac) {
          bool connected = mac->TraceConnectWithoutContext(
              "DlScheduling",
              MakeCallback(&MetricCollector::ReportDlScheduling, this));
          if (connected) {
            NS_LOG_UNCOND("Connected DlScheduling trace for eNodeB " << i);
          }
        }
      }
    }
  }
}

// 3rd arg is RSRP, 4th is SINR (linear), 5th is componentCarrierId
void MetricCollector::ReportUeSinr(uint16_t cellId, uint16_t rnti, double rsrp,
                                   double sinr, uint8_t componentCarrierId) {
  std::lock_guard<std::mutex> lock(m_mutex);
  auto it = m_rntiToUeIndex.find(rnti);
  if (it == m_rntiToUeIndex.end()) {
    return;
  }
  uint32_t ueIndex = it->second;
  m_ueMetrics[ueIndex].sumSinr += sinr;
  m_ueMetrics[ueIndex].sinrSamples++;
  // Debug log for verification (rarely needed, high volume)
}

void MetricCollector::ReportUeMeasurements(uint16_t rnti, uint16_t cellId,
                                           double rsrp, double rsrq,
                                           bool isServingCell,
                                           uint8_t componentCarrierId) {
  if (!isServingCell) {
    return;
  }

  std::lock_guard<std::mutex> lock(m_mutex);
  auto it = m_rntiToUeIndex.find(rnti);
  if (it == m_rntiToUeIndex.end()) {
    return;
  }
  uint32_t ueIndex = it->second;
  m_ueMetrics[ueIndex].sumRsrp += rsrp;
  m_ueMetrics[ueIndex].sumRsrpSq += rsrp * rsrp;
  m_ueMetrics[ueIndex].rsrpSamples++;
  m_ueMetrics[ueIndex].sumRsrq += rsrq;
  m_ueMetrics[ueIndex].sumRsrqSq += rsrq * rsrq;
  m_ueMetrics[ueIndex].rsrqSamples++;
  m_ueMetrics[ueIndex].servingCellId = cellId;
}

void MetricCollector::ReportUlPhyResourceBlocks(uint16_t rnti,
                                                const std::vector<int> &rbs) {
  std::lock_guard<std::mutex> lock(m_mutex);
  auto it = m_rntiToUeIndex.find(rnti);
  if (it == m_rntiToUeIndex.end()) {
    return;
  }
  uint32_t ueIndex = it->second;
  m_ueMetrics[ueIndex].ulRbCountSum += static_cast<uint32_t>(rbs.size());
  m_ueMetrics[ueIndex].ulRbSamples++;
}

void MetricCollector::ReportDlScheduling(DlSchedulingCallbackInfo info) {
  std::lock_guard<std::mutex> lock(m_mutex);
  uint16_t rnti = info.rnti;
  auto it = m_rntiToUeIndex.find(rnti);
  if (it == m_rntiToUeIndex.end()) {
    return;
  }
  uint32_t ueIndex = it->second;
  uint32_t tbs = info.sizeTb1 + info.sizeTb2;
  m_ueMetrics[ueIndex].bytesRx += tbs;

  // Estimate RBs from TBS: Rough heuristic for "Load"
  // Typ. 1 RB ~ 20-100 bytes depending on MCS.
  // We'll use a conservative divisor to get a non-zero "Index"
  uint32_t estimatedRbs = tbs / 50;
  if (estimatedRbs == 0 && tbs > 0) {
    estimatedRbs = 1;
  }
  m_ueMetrics[ueIndex].rbsAllocated += estimatedRbs;

  // VERIFY: Prove Throughput and RBs come from ns-3
  // std::cout << "VERIFY: DL_SCHED RNTI=" << rnti << " TBS=" << tbs
  //           << " EstRBs=" << estimatedRbs << std::endl;
}

void MetricCollector::ReportAppRx(uint32_t ueIndex, Ptr<const Packet> packet,
                                  const Address &from) {
  SeqTsHeader seqTs;
  // We need to copy because packet is const
  Ptr<Packet> p = packet->Copy();
  p->RemoveHeader(seqTs);
  Time txTime = seqTs.GetTs();
  Time delay = Simulator::Now() - txTime;

  std::lock_guard<std::mutex> lock(m_mutex);
  m_ueMetrics[ueIndex].sumLatency += delay.GetSeconds() * 1000.0; // ms
  m_ueMetrics[ueIndex].packetsRx++;

  // std::cout << "VERIFY: APP_RX RNTI=" << rnti
  //           << " Delay=" << delay.GetSeconds() * 1000.0 << "ms" << std::endl;
}

void MetricCollector::ReportHandoverStart(uint64_t imsi, uint16_t cellId,
                                          uint16_t rnti,
                                          uint16_t targetCellId) {
  std::lock_guard<std::mutex> lock(m_mutex);
  auto it = m_rntiToUeIndex.find(rnti);
  if (it == m_rntiToUeIndex.end()) {
    return;
  }
  uint32_t ueIndex = it->second;
  m_ueMetrics[ueIndex].handoverAttempts++;
}

void MetricCollector::ReportHandoverEndOk(uint64_t imsi, uint16_t cellId,
                                          uint16_t rnti) {
  std::lock_guard<std::mutex> lock(m_mutex);
  auto it = m_rntiToUeIndex.find(rnti);
  if (it == m_rntiToUeIndex.end()) {
    return;
  }
  uint32_t ueIndex = it->second;
  m_ueMetrics[ueIndex].handoverSuccesses++;
}

std::map<uint32_t, UeMetricAccumulator>
MetricCollector::GetAndResetUeMetrics() {
  std::map<uint32_t, UeMetricAccumulator> current;
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    current = m_ueMetrics;

    // Update packet loss from UdpServers
    for (const auto &[ueIndex, server] : m_ueUdpServers) {
      uint32_t totalLost = server->GetLost();

      // We can't access totalLost of non-existent RNTI (so it's safe)
      // But check if exists in m_lastLost
      if (m_lastLost.find(ueIndex) == m_lastLost.end()) {
        m_lastLost[ueIndex] = 0;
      }
      uint32_t delta = totalLost - m_lastLost[ueIndex];
      current[ueIndex].packetsLost = delta;
      m_lastLost[ueIndex] = totalLost;

      if (delta > 0) {
        // std::cout << "VERIFY: PKT_LOSS RNTI=" << rnti << " LossDelta=" <<
        // delta
        //           << std::endl;
      }
    }

    // Reset accumulators
    for (auto &kv : m_ueMetrics) {
      kv.second.Reset();
    }
  }
  return current;
}

class E2InterfaceManager : public SimpleRefCount<E2InterfaceManager> {
public:
  E2InterfaceManager(Ptr<LteHelper> lteHelper, NodeContainer enbNodes,
                     NodeContainer ueNodes, std::string brokers,
                     std::string topicSuffix = "");
  ~E2InterfaceManager();

  void EnableE2();
  void SetKpmInterval(Time interval);
  void SendKpmReport();
  void ProcessRcCommand(std::string command);

private:
  Ptr<LteHelper> m_lteHelper;
  NodeContainer m_enbNodes;
  NodeContainer m_ueNodes;
  std::string m_brokers;
  std::string m_topicSuffix;
  Time m_kpmInterval;
  EventId m_kpmEvent;
  // EventId m_pollEvent; // REMOVED: No more async polling

  // Kafka handles
  rd_kafka_t *m_producer;
  rd_kafka_t *m_consumer;
  rd_kafka_topic_t *m_kpmTopic;
  rd_kafka_topic_t *m_rcTopic;

  void SetupKafka();
  // Lock-step sync: blocks until Python sends an RC action.
  // Returns true if action received & applied, false on timeout.
  bool WaitForRcAction(int timeoutMs);

  // KPM metric collection
  struct UeMetrics {
    double throughputDl;
    double throughputUl;
    double delayDl;
    double packetLoss;
    double sinr;
    double rsrp;
    double rsrq;
    double ulRbAvg;
    uint32_t rbAllocated;
    double cqi;
    double rsrp_var;
    double rsrq_var;
    double bufferOccupancy;
    int32_t servingCellId;
    uint32_t handoverAttempts;
    uint32_t handoverSuccesses;
  };

  struct CellMetrics {
    uint32_t queueLength;
    double rbUtilization;
    double txPower;
    uint32_t numConnectedUes;
  };

  std::map<uint32_t, UeMetrics>
  CollectUeMetrics(const std::map<uint32_t, UeMetricAccumulator> &realMetrics);
  std::map<uint32_t, CellMetrics> CollectCellMetrics(
      const std::map<uint32_t, UeMetricAccumulator> &realMetrics);
};

E2InterfaceManager::E2InterfaceManager(Ptr<LteHelper> lteHelper,
                                       NodeContainer enbNodes,
                                       NodeContainer ueNodes,
                                       std::string brokers,
                                       std::string topicSuffix)
    : m_lteHelper(lteHelper), m_enbNodes(enbNodes), m_ueNodes(ueNodes),
      m_brokers(brokers), m_topicSuffix(topicSuffix),
      m_kpmInterval(MilliSeconds(100)), m_producer(nullptr),
      m_consumer(nullptr), m_kpmTopic(nullptr), m_rcTopic(nullptr) {}

E2InterfaceManager::~E2InterfaceManager() {
  if (m_kpmTopic) {
    rd_kafka_topic_destroy(m_kpmTopic);
  }
  if (m_producer) {
    rd_kafka_destroy(m_producer);
  }
  if (m_consumer) {
    rd_kafka_consumer_close(m_consumer);
    rd_kafka_destroy(m_consumer);
  }
}

void E2InterfaceManager::EnableE2() {
  NS_LOG_INFO("Enabling E2 interface via Kafka brokers: " << m_brokers);
  SetupKafka();

  // Create MetricCollector
  g_metricCollector = Create<MetricCollector>();
  g_metricCollector->RegisterUeTraces(m_ueNodes);
  g_metricCollector->RegisterEnbTraces(m_enbNodes);

  // Schedule periodic KPM reports
  m_kpmEvent = Simulator::Schedule(m_kpmInterval,
                                   &E2InterfaceManager::SendKpmReport, this);

  // Schedule periodic polling - REMOVED for Lock-Step
  // m_pollEvent = Simulator::Schedule(MilliSeconds(1),
  //                                   &E2InterfaceManager::PollKafka, this);

  NS_LOG_INFO("E2 lock-step mode enabled: Simulation will pause for actions.");
}

void E2InterfaceManager::SetKpmInterval(Time interval) {
  m_kpmInterval = interval;
}

void E2InterfaceManager::SetupKafka() {
  char errstr[512];
  rd_kafka_conf_t *conf;

  // --- Producer Setup (KPM Stream) ---
  conf = rd_kafka_conf_new();
  if (rd_kafka_conf_set(conf, "bootstrap.servers", m_brokers.c_str(), errstr,
                        sizeof(errstr)) != RD_KAFKA_CONF_OK) {
    NS_LOG_ERROR("Kafka Config Error: " << errstr);
    return;
  }

  m_producer = rd_kafka_new(RD_KAFKA_PRODUCER, conf, errstr, sizeof(errstr));
  if (!m_producer) {
    NS_LOG_ERROR("Failed to create producer: " << errstr);
    return;
  }

  std::string kpmTopicName = "e2_kpm_stream" + m_topicSuffix;
  m_kpmTopic = rd_kafka_topic_new(m_producer, kpmTopicName.c_str(), NULL);

  // --- Consumer Setup (RC Control) ---
  conf = rd_kafka_conf_new();
  if (rd_kafka_conf_set(conf, "bootstrap.servers", m_brokers.c_str(), errstr,
                        sizeof(errstr)) != RD_KAFKA_CONF_OK) {
    NS_LOG_ERROR("Kafka Config Error: " << errstr);
    return;
  }
  std::string groupId =
      "ns3-e2-agent" + m_topicSuffix + "_" + std::to_string(std::time(nullptr));
  rd_kafka_conf_set(conf, "group.id", groupId.c_str(), NULL, 0);
  rd_kafka_conf_set(conf, "auto.offset.reset", "latest", NULL, 0);

  m_consumer = rd_kafka_new(RD_KAFKA_CONSUMER, conf, errstr, sizeof(errstr));
  if (!m_consumer) {
    NS_LOG_ERROR("Failed to create consumer: " << errstr);
    return;
  }

  rd_kafka_poll_set_consumer(m_consumer);

  std::string rcTopicName = "e2_rc_control" + m_topicSuffix;
  rd_kafka_topic_partition_list_t *topics =
      rd_kafka_topic_partition_list_new(1);
  rd_kafka_topic_partition_list_add(topics, rcTopicName.c_str(),
                                    RD_KAFKA_PARTITION_UA);

  rd_kafka_resp_err_t err = rd_kafka_subscribe(m_consumer, topics);
  if (err) {
    NS_LOG_ERROR(
        "Failed to subscribe to e2_rc_control: " << rd_kafka_err2str(err));
  }

  rd_kafka_topic_partition_list_destroy(topics);

  NS_LOG_INFO("Kafka E2 Interface Initialized");
}

bool E2InterfaceManager::WaitForRcAction(int timeoutMs) {
  if (!m_consumer) {
    return false;
  }

  NS_LOG_INFO("Waiting for RC action from RL agent (sim_time="
              << Simulator::Now().GetSeconds() << "s, timeout=" << timeoutMs
              << "ms)...");

  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);

  while (std::chrono::steady_clock::now() < deadline) {
    // Service producer delivery reports while waiting
    if (m_producer) {
      rd_kafka_poll(m_producer, 0);
    }

    // Block-poll with short real-time timeout
    rd_kafka_message_t *rkm = rd_kafka_consumer_poll(m_consumer, 50);
    if (!rkm) {
      continue; // No message yet, keep waiting
    }

    if (rkm->err) {
      if (rkm->err != RD_KAFKA_RESP_ERR__PARTITION_EOF) {
        NS_LOG_WARN("Kafka consumer error while waiting for RC: "
                    << rd_kafka_message_errstr(rkm));
      }
      rd_kafka_message_destroy(rkm);
      continue;
    }

    // Got a valid message — drain any additional stale ones, keep LATEST
    std::string latestCommand((const char *)rkm->payload, rkm->len);
    rd_kafka_message_destroy(rkm);

    int drained = 1;
    rd_kafka_message_t *extra;
    while ((extra = rd_kafka_consumer_poll(m_consumer, 0)) != nullptr) {
      if (!extra->err) {
        latestCommand.assign((const char *)extra->payload, extra->len);
        drained++;
      }
      rd_kafka_message_destroy(extra);
    }

    if (drained > 1) {
      NS_LOG_INFO("WaitForRcAction: drained "
                  << drained << " queued messages, applying latest only");
    }

    // FORCE LOGGING (visible even without NS_LOG_INFO level)
    std::cout << "[LOCK-STEP] RC action received (sim_time="
              << Simulator::Now().GetSeconds() << "s): " << latestCommand
              << std::endl;

    ProcessRcCommand(latestCommand);
    return true;
  }

  NS_LOG_WARN("WaitForRcAction: TIMED OUT after "
              << timeoutMs
              << "ms — no RC action received. Continuing without update.");
  return false;
}

std::map<uint32_t, E2InterfaceManager::UeMetrics>
E2InterfaceManager::CollectUeMetrics(
    const std::map<uint32_t, UeMetricAccumulator> &realMetrics) {
  std::map<uint32_t, UeMetrics> metrics;

  for (uint32_t i = 0; i < m_ueNodes.GetN(); ++i) {
    Ptr<Node> ueNode = m_ueNodes.Get(i);
    // Get RNTI from UE device (checking first device is LTE)
    Ptr<LteUeNetDevice> ueDevice =
        ueNode->GetDevice(0)->GetObject<LteUeNetDevice>();
    if (!ueDevice) {
      continue;
    }

    uint16_t rnti = ueDevice->GetRrc()->GetRnti();
    if (g_metricCollector) {
      g_metricCollector->UpdateRntiMapping(i, rnti);
    }

    UeMetrics ueMetric;
    ueMetric.sinr = -10.0;
    ueMetric.throughputDl = 0.0;
    ueMetric.throughputUl = 0.0;
    ueMetric.delayDl = 0.0;
    ueMetric.packetLoss = 0.0;
    ueMetric.rbAllocated = 0;
    ueMetric.rsrp = -140.0;
    ueMetric.rsrq = -20.0;
    ueMetric.ulRbAvg = 0.0;
    ueMetric.cqi = 0.0;
    ueMetric.rsrp_var = 0.0;
    ueMetric.rsrq_var = 0.0;
    ueMetric.bufferOccupancy = 0.0;
    ueMetric.servingCellId = -1;
    ueMetric.handoverAttempts = 0;
    ueMetric.handoverSuccesses = 0;

    if (realMetrics.count(i)) {
      const auto &acc = realMetrics.at(i);
      if (acc.sinrSamples > 0) {
        ueMetric.sinr = 10 * log10(acc.sumSinr / acc.sinrSamples);
      }
      if (acc.rsrpSamples > 0) {
        ueMetric.rsrp = acc.sumRsrp / acc.rsrpSamples;
        double mean = ueMetric.rsrp;
        double meanSq = acc.sumRsrpSq / acc.rsrpSamples;
        double var = meanSq - (mean * mean);
        if (var < 1e-8 || var != var) // tiny or NaN
        {
          var = 0.0;
        }
        ueMetric.rsrp_var = var;
      }
      if (acc.rsrqSamples > 0) {
        ueMetric.rsrq = acc.sumRsrq / acc.rsrqSamples;
        double mean = ueMetric.rsrq;
        double meanSq = acc.sumRsrqSq / acc.rsrqSamples;
        double var = meanSq - (mean * mean);
        if (var < 1e-8 || var != var) {
          var = 0.0;
        }
        ueMetric.rsrq_var = var;
      }
      if (acc.ulRbSamples > 0) {
        ueMetric.ulRbAvg =
            static_cast<double>(acc.ulRbCountSum) / acc.ulRbSamples;
      }
      // Convert Bytes to Mbps (interval is important)
      double intervalSec = m_kpmInterval.GetSeconds();
      ueMetric.throughputDl = (acc.bytesRx * 8.0) / (intervalSec * 1e6);

      if (acc.packetsRx > 0) {
        ueMetric.delayDl = acc.sumLatency / acc.packetsRx;
      }

      // Calculate Packet Loss Ratio (PLR)
      double totalPackets = (double)(acc.packetsRx + acc.packetsLost);
      if (totalPackets > 0) {
        ueMetric.packetLoss = (double)acc.packetsLost / totalPackets;
      } else {
        ueMetric.packetLoss = 0.0;
      }

      ueMetric.rbAllocated = acc.rbsAllocated;
      // Placeholder buffer occupancy (not directly available); keep zero for
      // now
      ueMetric.bufferOccupancy = 0.0;
      ueMetric.servingCellId = acc.servingCellId;

      // Estimate CQI from SINR (simple linear mapping clamped to 0-15)
      double sinrDb = ueMetric.sinr;
      int estCqi = static_cast<int>(std::round((sinrDb + 10.0) / 1.5));
      if (estCqi < 0) {
        estCqi = 0;
      }
      if (estCqi > 15) {
        estCqi = 15;
      }
      ueMetric.cqi = static_cast<double>(estCqi);

      // Handover metrics
      ueMetric.handoverAttempts = acc.handoverAttempts;
      ueMetric.handoverSuccesses = acc.handoverSuccesses;
    }

    metrics[i] = ueMetric;
  }

  return metrics;
}

std::map<uint32_t, E2InterfaceManager::CellMetrics>
E2InterfaceManager::CollectCellMetrics(
    const std::map<uint32_t, UeMetricAccumulator> &realMetrics) {
  std::map<uint32_t, CellMetrics> metrics;

  // Pre-compute per-cell UE counts, RB usage, and queue from UE metrics
  uint32_t numCells = m_enbNodes.GetN();
  std::vector<uint32_t> uesPerCell(numCells, 0);
  std::vector<uint64_t> rbsPerCell(numCells, 0);
  std::vector<double> queuePerCell(numCells, 0.0);

  for (uint32_t ueIdx = 0; ueIdx < m_ueNodes.GetN(); ++ueIdx) {
    if (realMetrics.count(ueIdx)) {
      const auto &acc = realMetrics.at(ueIdx);
      int32_t cellId = acc.servingCellId;
      if (cellId >= 0 && cellId < (int32_t)numCells) {
        uesPerCell[cellId]++;
        rbsPerCell[cellId] += acc.rbsAllocated;
        // Estimate queue from UE buffer occupancy (bytes pending)
        queuePerCell[cellId] +=
            static_cast<double>(acc.bytesRx > 0 ? acc.packetsLost : 0);
      }
    }
  }

  // Dynamic: read actual bandwidth from first eNB device (fallback 50 for 10
  // MHz)
  double totalRbsPerInterval = 50.0;
  if (m_enbNodes.GetN() > 0) {
    Ptr<LteEnbNetDevice> dev0 =
        m_enbNodes.Get(0)->GetDevice(0)->GetObject<LteEnbNetDevice>();
    if (dev0) {
      totalRbsPerInterval = static_cast<double>(dev0->GetDlBandwidth());
    }
  }

  for (uint32_t i = 0; i < numCells; ++i) {
    Ptr<Node> enbNode = m_enbNodes.Get(i);
    Ptr<LteEnbNetDevice> enbLteDevice =
        enbNode->GetDevice(0)->GetObject<LteEnbNetDevice>();
    if (!enbLteDevice) {
      continue;
    }

    Ptr<LteEnbPhy> enbPhy = enbLteDevice->GetPhy();

    CellMetrics cellMetric;
    cellMetric.txPower = enbPhy->GetTxPower();
    cellMetric.numConnectedUes = uesPerCell[i];
    cellMetric.queueLength = queuePerCell[i];
    // RB utilization: allocated RBs / total available (clamped 0-1)
    cellMetric.rbUtilization =
        (totalRbsPerInterval > 0)
            ? std::min(1.0,
                       static_cast<double>(rbsPerCell[i]) / totalRbsPerInterval)
            : 0.0;

    metrics[i] = cellMetric;
  }

  return metrics;
}

void E2InterfaceManager::SendKpmReport() {
  if (!m_producer || !m_kpmTopic) {
    m_kpmEvent = Simulator::Schedule(m_kpmInterval,
                                     &E2InterfaceManager::SendKpmReport, this);
    return;
  }

  // Collect metrics (Fetch ONCE to ensure consistency between UE and Cell
  // metrics)
  std::map<uint32_t, UeMetricAccumulator> realMetrics;
  if (g_metricCollector) {
    realMetrics = g_metricCollector->GetAndResetUeMetrics();
  }

  auto ueMetrics = CollectUeMetrics(realMetrics);
  auto cellMetrics = CollectCellMetrics(realMetrics);

  // Build JSON KPM report
  std::stringstream kpmJson;
  kpmJson << "{";
  kpmJson << "\"timestamp\":" << Simulator::Now().GetSeconds() << ",";

  // UE metrics
  for (const auto &[ueId, metrics] : ueMetrics) {
    kpmJson << "\"ue_" << ueId << "_tput\":" << metrics.throughputDl << ",";
    kpmJson << "\"ue_" << ueId << "_delay\":" << metrics.delayDl << ",";
    kpmJson << "\"ue_" << ueId << "_loss\":" << metrics.packetLoss << ",";
    kpmJson << "\"ue_" << ueId << "_sinr\":" << metrics.sinr << ",";
    kpmJson << "\"ue_" << ueId << "_rsrp\":" << metrics.rsrp << ",";
    kpmJson << "\"ue_" << ueId << "_rsrq\":" << metrics.rsrq << ",";
    kpmJson << "\"ue_" << ueId << "_ul_rbs\":" << metrics.ulRbAvg << ",";
    kpmJson << "\"ue_" << ueId << "_rbs\":" << metrics.rbAllocated << ",";
    kpmJson << "\"ue_" << ueId << "_cqi\":" << metrics.cqi << ",";
    kpmJson << "\"ue_" << ueId << "_rsrp_var\":" << metrics.rsrp_var << ",";
    kpmJson << "\"ue_" << ueId << "_rsrq_var\":" << metrics.rsrq_var << ",";
    kpmJson << "\"ue_" << ueId << "_buffer\":" << metrics.bufferOccupancy
            << ",";
    kpmJson << "\"ue_" << ueId << "_cell\":" << metrics.servingCellId << ",";
    kpmJson << "\"ue_" << ueId << "_ho_att\":" << metrics.handoverAttempts
            << ",";
    kpmJson << "\"ue_" << ueId << "_ho_succ\":" << metrics.handoverSuccesses
            << ",";
  }

  // Prepare per-cell aggregates (avg rb request, load)
  uint32_t numCells = m_enbNodes.GetN();
  std::vector<double> sumRbReq(numCells, 0.0);
  std::vector<uint32_t> countUes(numCells, 0);
  for (const auto &[ueId, metrics] : ueMetrics) {
    // Use actual serving cell instead of modulo
    int32_t cellId = metrics.servingCellId;
    if (cellId >= 0 && static_cast<uint32_t>(cellId) < numCells) {
      sumRbReq[cellId] += static_cast<double>(metrics.rbAllocated);
      countUes[cellId] += 1;
    }
  }

  // Cell metrics
  for (const auto &[cellId, metrics] : cellMetrics) {
    kpmJson << "\"cell_" << cellId << "_queue\":" << metrics.queueLength << ",";
    kpmJson << "\"cell_" << cellId << "_rb_util\":" << metrics.rbUtilization
            << ",";
    kpmJson << "\"cell_" << cellId << "_power\":" << metrics.txPower << ",";
    kpmJson << "\"cell_" << cellId << "_ues\":" << metrics.numConnectedUes
            << ",";

    double avgRbReq = 0.0;
    if (cellId < sumRbReq.size() && countUes[cellId] > 0) {
      avgRbReq = sumRbReq[cellId] / static_cast<double>(countUes[cellId]);
    }
    double cellLoad = static_cast<double>(countUes[cellId]);

    kpmJson << "\"cell_" << cellId << "_load\":" << cellLoad << ",";
    kpmJson << "\"cell_" << cellId << "_avg_rb_req\":" << avgRbReq;

    if (cellId < cellMetrics.size() - 1) {
      kpmJson << ",";
    }
  }

  kpmJson << "}";

  std::string kpmString = kpmJson.str();

  // Send KPM via Kafka with retry on transient failure
  bool kpmSent = false;
  for (int attempt = 0; attempt < 3 && !kpmSent; ++attempt) {
    if (rd_kafka_produce(m_kpmTopic, RD_KAFKA_PARTITION_UA, RD_KAFKA_MSG_F_COPY,
                         (void *)kpmString.c_str(), kpmString.length(), NULL, 0,
                         NULL) == -1) {
      NS_LOG_WARN("Failed to produce KPM report (attempt " << attempt + 1
                                                           << ")");
      if (rd_kafka_last_error() == RD_KAFKA_RESP_ERR__QUEUE_FULL) {
        rd_kafka_poll(m_producer, 500); // Drain delivery reports, then retry
      }
    } else {
      kpmSent = true;
    }
  }

  if (kpmSent) {
    // Flush to guarantee KPM reaches broker before we block-wait
    rd_kafka_flush(m_producer, 5000);
    NS_LOG_DEBUG("Sent + flushed KPM report (sim_time="
                 << Simulator::Now().GetSeconds() << "s)");

    // LOCK-STEP: Pause simulation until RL agent responds with an action.
    // Timeout 60s matches Python's max_wait_s=60.0.
    WaitForRcAction(60000);
  } else {
    NS_LOG_ERROR(
        "KPM produce failed after 3 attempts — skipping WaitForRcAction "
        "to avoid deadlock");
  }

  // Schedule next report
  m_kpmEvent = Simulator::Schedule(m_kpmInterval,
                                   &E2InterfaceManager::SendKpmReport, this);
}

void E2InterfaceManager::ProcessRcCommand(std::string command) {
  NS_LOG_INFO("Processing RC command: " << command);

  // JSON format (flat keys from Python):
  //   {"cell_0": 1, "cell_0_tx_power_dbm": 46.0, "cell_0_hysteresis_db": 1.2,
  //   ...}
  // We search for "cell_X" to find the cell, then look for "cell_X_<param>"
  // keys.

  for (uint32_t i = 0; i < m_enbNodes.GetN(); ++i) {
    std::stringstream ss;
    ss << "\"cell_" << i << "\"";
    std::string cellKey = ss.str(); // e.g. "cell_0"

    size_t cellPos = command.find(cellKey);
    if (cellPos == std::string::npos) {
      continue;
    }

    // Build per-cell key prefix for flat JSON: "cell_0_tx_power_dbm" etc.
    std::string cellPfx = "cell_" + std::to_string(i) + "_";

    auto parseValue = [&](const std::string &paramName,
                          double &outValue) -> bool {
      std::string key = cellPfx + paramName;
      size_t keyPos = command.find(key);
      if (keyPos == std::string::npos) {
        return false;
      }

      // Found the key part (e.g. cell_0_tx_power_dbm)
      // Now find the colon after it
      size_t colonPos = command.find(":", keyPos + key.length());
      if (colonPos == std::string::npos) {
        return false;
      }

      size_t valStart = colonPos + 1;
      // Skip whitespace and possible negative sign
      while (valStart < command.length() &&
             (command[valStart] == ' ' || command[valStart] == '\t')) {
        valStart++;
      }

      size_t valEnd = valStart;
      if (valEnd < command.length() && command[valEnd] == '-') {
        valEnd++; // allow negative
      }
      while (valEnd < command.length() &&
             (isdigit(command[valEnd]) || command[valEnd] == '.')) {
        valEnd++;
      }

      if (valEnd <= valStart) {
        return false;
      }

      std::string valStr = command.substr(valStart, valEnd - valStart);
      try {
        outValue = std::stod(valStr);
        return true;
      } catch (...) {
        NS_LOG_WARN("Failed to parse value for " << key << ": " << valStr);
        return false;
      }
    };

    Ptr<Node> enbNode = m_enbNodes.Get(i);
    Ptr<LteEnbNetDevice> dev =
        enbNode->GetDevice(0)->GetObject<LteEnbNetDevice>();
    if (!dev) {
      continue;
    }

    double txPower = 0.0;
    if (parseValue("tx_power_dbm", txPower)) {
      // Clamp to safe LTE range 10–46 dBm
      txPower = std::max(10.0, std::min(46.0, txPower));
      dev->GetPhy()->SetTxPower(txPower);
      std::cout << "Set Cell " << i << " TxPower to " << txPower << " dBm"
                << std::endl;
    }

    // 2. CIO: Neighbor Cell Offset (how this cell is seen by others)
    // Positive value -> Pulls UEs to this cell from neighbors.
    double cio = 0.0;
    if (parseValue("cell_individual_offset_db", cio)) {
      std::cout << "Cell " << i << " RC command: Received CIO = " << cio
                << std::endl;
      // 3GPP Q-OffsetRange is even numbers [-24..24]
      int8_t cio_int = static_cast<int8_t>(std::round(cio / 2.0) * 2.0);
      cio_int = std::max((int8_t)-24, std::min((int8_t)24, cio_int));

      uint16_t thisPhysCellId = dev->GetCellId();
      bool applied = false;
      uint32_t neighborsFoundCount = 0;

      // CIO logic: Update this cell's offset in ALL OTHER cells' neighbor lists
      for (uint32_t j = 0; j < m_enbNodes.GetN(); ++j) {
        if (i == j)
          continue; // Don't set self-offset

        Ptr<LteEnbRrc> otherRrc = m_enbNodes.Get(j)
                                      ->GetDevice(0)
                                      ->GetObject<LteEnbNetDevice>()
                                      ->GetRrc();
        LteRrcSap::MeasConfig &otherMeasConfig =
            PeekPointer(otherRrc)->*get(LteEnbRrcMeasConfigTag());

        bool foundNeighbor = false;
        if (!otherMeasConfig.measObjectToAddModList.empty()) {
          // Add to the first EUTRA meas object found
          auto &measObj =
              otherMeasConfig.measObjectToAddModList.front().measObjectEutra;

          for (auto &cellMod : measObj.cellsToAddModList) {
            if (cellMod.physCellId == thisPhysCellId) {
              cellMod.cellIndividualOffset = cio_int;
              foundNeighbor = true;
              neighborsFoundCount++;
            }
          }

          if (!foundNeighbor) {
            // Add as new entry in existing meas object
            LteRrcSap::CellsToAddMod newCell;
            newCell.physCellId = thisPhysCellId;
            newCell.cellIndividualOffset = cio_int;
            measObj.cellsToAddModList.push_back(newCell);
            foundNeighbor = true;
            neighborsFoundCount++;
          }
        }

        if (foundNeighbor) {
          applied = true;
          // Trigger reconfiguration for all UEs in this neighbor cell
          std::map<uint16_t, Ptr<UeManager>> &ueMap =
              PeekPointer(otherRrc)->*get(LteEnbRrcUeMapTag());
          for (auto const &[rnti, ueMgr] : ueMap) {
            (PeekPointer(ueMgr)->*get(UeManagerReconfigTag()))();
          }
        }
      }

      if (applied) {
        std::cout << "Action Applied: Set Cell " << i << " CIO to "
                  << (int)cio_int << " dB (in " << neighborsFoundCount
                  << " neighbor lists)" << std::endl;
      } else {
        std::cout << "Cell " << i
                  << " RC command: CIO not applied (this cell not found in Any "
                     "neighbor lists)."
                  << std::endl;
      }
    }

    // 3. TimeToTrigger Control (Serving Cell Parameter)
    double ttt = 0.0;
    if (parseValue("time_to_trigger_ms", ttt)) {
      uint16_t ttt_uint = static_cast<uint16_t>(ttt);
      Ptr<LteEnbRrc> rrc = dev->GetRrc();
      LteRrcSap::MeasConfig &measConfig =
          PeekPointer(rrc)->*get(LteEnbRrcMeasConfigTag());

      bool updated = false;
      for (auto &reportCfgMod : measConfig.reportConfigToAddModList) {
        if (reportCfgMod.reportConfigEutra.eventId ==
            LteRrcSap::ReportConfigEutra::EVENT_A3) {
          reportCfgMod.reportConfigEutra.timeToTrigger = ttt_uint;
          updated = true;
        }
      }

      if (updated) {
        // Trigger reconfiguration for all UEs in this cell
        std::map<uint16_t, Ptr<UeManager>> &ueMap =
            PeekPointer(rrc)->*get(LteEnbRrcUeMapTag());
        for (auto const &[rnti, ueMgr] : ueMap) {
          (PeekPointer(ueMgr)->*get(UeManagerReconfigTag()))();
        }
        std::cout << "Action Applied: Set Cell " << i << " TTT to " << ttt_uint
                  << " ms" << std::endl;
      }

      // Also update HandoverAlgorithm attribute for consistency (if it exists)
      Ptr<LteHandoverAlgorithm> hoAlgo =
          enbNode->GetObject<LteHandoverAlgorithm>();
      if (hoAlgo) {
        hoAlgo->SetAttribute("TimeToTrigger",
                             TimeValue(MilliSeconds(ttt_uint)));
      }
    }
  }
}

// ============================================================================
// Congestion Scenario Generator
// ============================================================================

class CongestionScenarioManager
    : public SimpleRefCount<CongestionScenarioManager> {
public:
  enum ScenarioType {
    FLASH_CROWD,        // Sudden influx of users in one cell
    MOBILITY_STORM,     // High-speed users moving across cells
    TRAFFIC_BURST,      // Periodic surges in application data
    HANDOVER_PING_PONG, // Users oscillating between cell boundaries
    SLEEPY_CAMPUS,      // Energy optimization (Low vs High traffic)
    AMBULANCE,          // QoS prioritization for emergency streams
    ADVERSARIAL,        // Reliability testing under signal fluctuations
    COMMUTER_RUSH,      // Mass group handover (50+ UEs)
    MIXED_REALITY,      // Network slicing (Latency-sensitive VR vs Bulk TCP)
    URBAN_CANYON,       // PHY blockage and beamforming recovery
    IOT_TSUNAMI,        // Massive device scale (100+ UEs)
    SPECTRUM_CRUNCH     // Resource management with Carrier Aggregation
  };

  CongestionScenarioManager(NodeContainer ueNodes, Ptr<LteHelper> lteHelper,
                            ApplicationContainer clientApps);
  void ActivateScenario(ScenarioType type, Time startTime);

private:
  NodeContainer m_ueNodes;
  Ptr<LteHelper> m_lteHelper;
  ApplicationContainer m_clientApps;

  void TriggerFlashCrowd(uint32_t numUes);
  void TriggerMobilityStorm();
  void TriggerTrafficBurst();
  void TriggerHandoverPingPong();
  void TriggerSleepyCampus(bool highLoad);
  void TriggerAmbulance();
  void TriggerAdversarial(bool active);
  void TriggerCommuterRush();
  void TriggerMixedReality();
  void TriggerUrbanCanyon();
  void TriggerIotTsunami();
  void TriggerSpectrumCrunch();
};

CongestionScenarioManager::CongestionScenarioManager(
    NodeContainer ueNodes, Ptr<LteHelper> lteHelper,
    ApplicationContainer clientApps)
    : m_ueNodes(ueNodes), m_lteHelper(lteHelper), m_clientApps(clientApps) {}

void CongestionScenarioManager::ActivateScenario(ScenarioType type,
                                                 Time startTime) {
  switch (type) {
  case FLASH_CROWD:
    Simulator::Schedule(startTime,
                        &CongestionScenarioManager::TriggerFlashCrowd, this,
                        m_ueNodes.GetN() / 2);
    break;
  case MOBILITY_STORM:
    Simulator::Schedule(startTime,
                        &CongestionScenarioManager::TriggerMobilityStorm, this);
    break;
  case TRAFFIC_BURST:
    Simulator::Schedule(startTime,
                        &CongestionScenarioManager::TriggerTrafficBurst, this);
    break;
  case HANDOVER_PING_PONG:
    Simulator::Schedule(
        startTime, &CongestionScenarioManager::TriggerHandoverPingPong, this);
    break;
  case SLEEPY_CAMPUS:
    Simulator::Schedule(
        startTime, &CongestionScenarioManager::TriggerSleepyCampus, this, true);
    break;
  case AMBULANCE:
    Simulator::Schedule(startTime, &CongestionScenarioManager::TriggerAmbulance,
                        this);
    break;
  case COMMUTER_RUSH:
    Simulator::Schedule(startTime,
                        &CongestionScenarioManager::TriggerCommuterRush, this);
    break;
  case MIXED_REALITY:
    Simulator::Schedule(startTime,
                        &CongestionScenarioManager::TriggerMixedReality, this);
    break;
  case URBAN_CANYON:
    Simulator::Schedule(startTime,
                        &CongestionScenarioManager::TriggerUrbanCanyon, this);
    break;
  case IOT_TSUNAMI:
    Simulator::Schedule(startTime,
                        &CongestionScenarioManager::TriggerIotTsunami, this);
    break;
  case SPECTRUM_CRUNCH:
    Simulator::Schedule(
        startTime, &CongestionScenarioManager::TriggerSpectrumCrunch, this);
    break;
  case ADVERSARIAL:
    Simulator::Schedule(
        startTime, &CongestionScenarioManager::TriggerAdversarial, this, true);
    break;
  }
}

void CongestionScenarioManager::TriggerFlashCrowd(uint32_t numUes) {
  NS_LOG_INFO("TRIGGERING FLASH CROWD: "
              << numUes << " UEs suddenly requesting high bandwidth");
  // Increase traffic for the first numUes
  for (uint32_t i = 0; i < numUes && i < m_clientApps.GetN(); ++i) {
    Ptr<UdpClient> app = m_clientApps.Get(i)->GetObject<UdpClient>();
    if (app) {
      // Increase rate: 1024 bytes every 2ms = ~4 Mbps
      // Using 500us to generate ~16Mbps per UE to force congestion
      app->SetAttribute("Interval", TimeValue(MicroSeconds(500)));
    }
  }
}

void CongestionScenarioManager::TriggerMobilityStorm() {
  NS_LOG_INFO(
      "TRIGGERING MOBILITY STORM: Rapid UE movements causing handover cascade");
  for (uint32_t i = 0; i < m_ueNodes.GetN(); ++i) {
    Ptr<Node> ueNode = m_ueNodes.Get(i);
    Ptr<MobilityModel> mobility = ueNode->GetObject<MobilityModel>();

    if (Ptr<ConstantVelocityMobilityModel> cvMobility =
            DynamicCast<ConstantVelocityMobilityModel>(mobility)) {
      Vector velocity = cvMobility->GetVelocity();
      cvMobility->SetVelocity(Vector(velocity.x * 5, velocity.y * 5, 0));
    }
  }
}

void CongestionScenarioManager::TriggerTrafficBurst() {
  NS_LOG_INFO("TRIGGERING TRAFFIC BURST: Sudden spike in data transmission");
  // Spike for ALL UEs
  for (uint32_t i = 0; i < m_clientApps.GetN(); ++i) {
    Ptr<UdpClient> app = m_clientApps.Get(i)->GetObject<UdpClient>();
    if (app) {
      // Aggressive burst to force congestion
      app->SetAttribute("Interval", TimeValue(MicroSeconds(500)));
    }
  }
}

void CongestionScenarioManager::TriggerHandoverPingPong() {
  NS_LOG_INFO("TRIGGERING HANDOVER PING-PONG: UEs oscillating between cells");
  // Force UEs to move rapidly
  for (uint32_t i = 0; i < m_ueNodes.GetN(); ++i) {
    Ptr<MobilityModel> mobility = m_ueNodes.Get(i)->GetObject<MobilityModel>();
    if (Ptr<ConstantVelocityMobilityModel> cv =
            DynamicCast<ConstantVelocityMobilityModel>(mobility)) {
      cv->SetVelocity(Vector(50.0, 0.0, 0.0)); // High speed
    }
  }
}

void CongestionScenarioManager::TriggerSleepyCampus(bool highLoad) {
  NS_LOG_INFO("TRIGGERING SLEEPY CAMPUS: HighLoad=" << highLoad);
  for (uint32_t i = 0; i < m_clientApps.GetN(); ++i) {
    if (Ptr<UdpClient> app = m_clientApps.Get(i)->GetObject<UdpClient>()) {
      app->SetAttribute("Interval",
                        TimeValue(MicroSeconds(highLoad ? 500 : 5000)));
    }
  }
  Simulator::Schedule(Seconds(3.0),
                      &CongestionScenarioManager::TriggerSleepyCampus, this,
                      !highLoad);
}

void CongestionScenarioManager::TriggerAmbulance() {
  NS_LOG_INFO("TRIGGERING AMBULANCE: Priority User Injection");
  if (m_clientApps.GetN() > 0) {
    Ptr<UdpClient> app =
        m_clientApps.Get(m_clientApps.GetN() - 1)->GetObject<UdpClient>();
    if (app) {
      app->SetAttribute("Interval", TimeValue(MicroSeconds(100)));
      app->SetAttribute("PacketSize", UintegerValue(1400));
    }
  }
}

void CongestionScenarioManager::TriggerAdversarial(bool active) {
  NS_LOG_INFO("TRIGGERING ADVERSARIAL: Noise=" << active);
  Config::Set("/NodeList/*/DeviceList/*/LteEnbPhy/NoiseFigure",
              DoubleValue(active ? 20.0 : 5.0));
  Simulator::Schedule(Seconds(4.0),
                      &CongestionScenarioManager::TriggerAdversarial, this,
                      !active);
}

void CongestionScenarioManager::TriggerCommuterRush() {
  NS_LOG_INFO("TRIGGERING COMMUTER RUSH: Mass Handover Event");
  // Move all UEs rapidly to the right (simulating a train)
  for (uint32_t i = 0; i < m_ueNodes.GetN(); ++i) {
    Ptr<MobilityModel> mobility = m_ueNodes.Get(i)->GetObject<MobilityModel>();
    if (Ptr<ConstantVelocityMobilityModel> cv =
            DynamicCast<ConstantVelocityMobilityModel>(mobility)) {
      cv->SetVelocity(Vector(200.0, 0.0, 0.0)); // Very fast movement
    }
  }
}

void CongestionScenarioManager::TriggerMixedReality() {
  NS_LOG_INFO("TRIGGERING MIXED REALITY: Slicing Contention");
  // Split UEs into VR (Slice A) and Download (Slice B)
  // VR: Low latency, small interval. DL: High bandwidth.
  for (uint32_t i = 0; i < m_clientApps.GetN(); ++i) {
    Ptr<UdpClient> app = m_clientApps.Get(i)->GetObject<UdpClient>();
    if (!app) {
      continue;
    }

    if (i % 2 == 0) {
      // Slice A (VR): 1500 bytes every 2ms
      app->SetAttribute("Interval", TimeValue(MilliSeconds(2)));
      app->SetAttribute("PacketSize", UintegerValue(1500));
    } else {
      // Slice B (Download/Bulk): 1400 bytes every 1ms
      app->SetAttribute("Interval", TimeValue(MilliSeconds(1)));
      app->SetAttribute("PacketSize", UintegerValue(1400));
    }
  }
}

void CongestionScenarioManager::TriggerUrbanCanyon() {
  NS_LOG_INFO("TRIGGERING URBAN CANYON: Blockage Event");
  // Simulate drop in signal by increasing noise significantly for a short
  // duration "Blockage" model is complex to swap runtime, simulating via
  // degradations
  Config::Set("/NodeList/*/DeviceList/*/LteEnbPhy/NoiseFigure",
              DoubleValue(25.0));

  // Recovery after 2 seconds
  Simulator::Schedule(
      Seconds(2.0), +[](void) {
        Config::Set("/NodeList/*/DeviceList/*/LteEnbPhy/NoiseFigure",
                    DoubleValue(5.0));
      });
}

void CongestionScenarioManager::TriggerIotTsunami() {
  NS_LOG_INFO("TRIGGERING IOT TSUNAMI: Massive Control Plane Load");
  // Small packets, frequent intervals for ALL UEs
  for (uint32_t i = 0; i < m_clientApps.GetN(); ++i) {
    Ptr<UdpClient> app = m_clientApps.Get(i)->GetObject<UdpClient>();
    if (app) {
      app->SetAttribute("PacketSize", UintegerValue(40)); // Small packet (M2M)
      app->SetAttribute(
          "Interval",
          TimeValue(MilliSeconds(1))); // 1000 pps to flood scheduler
    }
  }
}

void CongestionScenarioManager::TriggerSpectrumCrunch() {
  NS_LOG_INFO("TRIGGERING SPECTRUM CRUNCH: Simulated CA Load");
  // Increase load significantly to force need for secondary carrier
  for (uint32_t i = 0; i < m_clientApps.GetN(); ++i) {
    Ptr<UdpClient> app = m_clientApps.Get(i)->GetObject<UdpClient>();
    if (app) {
      app->SetAttribute(
          "Interval",
          TimeValue(MicroSeconds(100))); // Ultra high load (10k pps)
    }
  }
}

ApplicationContainer SetupTraffic(NodeContainer ues, NodeContainer remoteHost,
                                  Ptr<LteHelper> lteHelper) {
  // Install UDP traffic on all UEs
  // UEs receive data from Remote Host (DL)
  uint16_t dlPort = 1234;

  // Remote Host is traffic source
  ApplicationContainer serverApps;
  ApplicationContainer clientApps;

  for (uint32_t i = 0; i < ues.GetN(); ++i) {
    Ptr<Node> ue = ues.Get(i);
    Ptr<NetDevice> ueDevice = ue->GetDevice(0);
    Ipv4Address ueAddr = ue->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();

    std::stringstream ss;
    ueAddr.Print(ss);
    NS_LOG_UNCOND("SetupTraffic: UE " << i << " IP: " << ss.str());

    // Source: Remote Host sending to UE
    UdpClientHelper dlClient(ueAddr, dlPort);
    dlClient.SetAttribute("Interval", TimeValue(MilliSeconds(20))); // 50 pps
    dlClient.SetAttribute("MaxPackets", UintegerValue(1000000));
    dlClient.SetAttribute("PacketSize", UintegerValue(1024));

    serverApps.Add(dlClient.Install(remoteHost));

    // Sink: UE receiving
    UdpServerHelper dlServer(dlPort);
    clientApps.Add(dlServer.Install(ue));
  }

  serverApps.Start(Seconds(0.1));
  clientApps.Start(Seconds(0.1));

  return serverApps;
}

// ============================================================================
// Main Simulation
// ============================================================================

int main(int argc, char *argv[]) {
  // Simulation parameters
  uint32_t numUes = 20;
  uint32_t numCells = 3;
  double simTime = 10.0;
  uint32_t seed = 42;
  uint32_t kpmInterval = 100; // milliseconds
  bool enableE2 = true;
  std::string congestionScenario = "flash_crowd";
  std::string kafkaBrokers = "localhost:9092";
  uint32_t bandwidthRbs = 25;   // Default 5 MHz (25 RBs)
  std::string topicSuffix = ""; // Kafka topic suffix for parallel env isolation

  // Enable Realtime Simulator for RL synchronization
  GlobalValue::Bind("SimulatorImplementationType",
                    StringValue("ns3::RealtimeSimulatorImpl"));

  CommandLine cmd;
  cmd.AddValue("numUes", "Number of UEs", numUes);
  cmd.AddValue("numCells", "Number of eNB/gNB cells", numCells);
  cmd.AddValue("simTime", "Total simulation time (seconds)", simTime);
  cmd.AddValue("seed", "Random seed", seed);
  cmd.AddValue("kpmInterval", "KPM reporting interval (ms)", kpmInterval);
  cmd.AddValue("enableE2", "Enable E2 interface", enableE2);
  cmd.AddValue("scenario", "Congestion scenario type", congestionScenario);
  cmd.AddValue("kafkaBrokers", "Kafka bootstrap servers", kafkaBrokers);
  cmd.AddValue("bandwidthRbs", "DL/UL Bandwidth in RBs (e.g. 25, 50, 100)",
               bandwidthRbs);
  cmd.AddValue("topicSuffix",
               "Kafka topic suffix for parallel environments (e.g. _0, _1)",
               topicSuffix);
  cmd.Parse(argc, argv);

  try {
    // Scenario-based overrides (Must happen before node creation)
    if (congestionScenario == "iot_tsunami") {
      NS_LOG_INFO("Configuring IoT Tsunami: Increasing UEs to 100 and changing "
                  "traffic model");
      numUes = 100;
    } else if (congestionScenario == "spectrum_crunch") {
      NS_LOG_INFO(
          "Configuring Spectrum Crunch: Ensuring High Bandwidth (100 RBs)");
      bandwidthRbs = 100;
    }

    // Set random seed
    RngSeedManager::SetSeed(seed);

    NS_LOG_INFO(
        "=== Radio-Cortex O-RAN Congestion Scenario (Kafka Native) ===");
    NS_LOG_INFO("UEs: " << numUes << ", Cells: " << numCells);
    NS_LOG_INFO("Simulation time: " << simTime << "s");

    // Explicitly enable logging for this scenario to fix "empty logs" issue
    LogComponentEnable("RadioCortexOranScenario", LOG_LEVEL_ALL);

    // Global LTE defaults (must be set before helper creation)
    Config::SetDefault("ns3::LteHelper::UseIdealRrc", BooleanValue(true));
    Config::SetDefault("ns3::LteHelper::HandoverAlgorithm",
                       StringValue("ns3::A3RsrpHandoverAlgorithm"));

    // Create LTE/EPC helpers
    Ptr<LteHelper> lteHelper = CreateObject<LteHelper>();
    Ptr<PointToPointEpcHelper> epcHelper =
        CreateObject<PointToPointEpcHelper>();
    lteHelper->SetEpcHelper(epcHelper);

    // Stabilize RRC: avoid UE manager asserts from rapid RRC churn
    lteHelper->SetAttribute("UseIdealRrc", BooleanValue(true));
    lteHelper->SetHandoverAlgorithmType("ns3::A3RsrpHandoverAlgorithm");
    lteHelper->SetHandoverAlgorithmAttribute("Hysteresis", DoubleValue(3.0));
    lteHelper->SetHandoverAlgorithmAttribute("TimeToTrigger",
                                             TimeValue(MilliSeconds(256)));

    // Set Scheduler
    lteHelper->SetSchedulerType("ns3::PfFfMacScheduler"); // Proportional Fair

    // Scenario: Spectrum Crunch (Carrier Aggregation)
    if (congestionScenario == "spectrum_crunch") {
      NS_LOG_INFO(
          "Configuring Spectrum Crunch: Enabling Carrier Aggregation (2 CCs)");
      lteHelper->SetAttribute("UseCa", BooleanValue(true));
      lteHelper->SetAttribute("NumberOfComponentCarriers", UintegerValue(2));
      lteHelper->SetAttribute("EnbComponentCarrierManager",
                              StringValue("ns3::RrComponentCarrierManager"));
    }

    // Scenario-Specific Channel Configuration
    if (congestionScenario == "urban_canyon") {
      NS_LOG_INFO("Configuring Urban Canyon: Using "
                  "LogDistancePropagationLossModel (Exponent 3.8)");
      lteHelper->SetAttribute(
          "PathlossModel", StringValue("ns3::LogDistancePropagationLossModel"));
      lteHelper->SetPathlossModelAttribute("Exponent", DoubleValue(3.8));
    } else {
      // Default simple pathloss
      lteHelper->SetAttribute("PathlossModel",
                              StringValue("ns3::FriisPropagationLossModel"));
    }

    // Set Configurable Bandwidth
    NS_LOG_INFO("Configuring Bandwidth: " << bandwidthRbs << " RBs");
    lteHelper->SetEnbDeviceAttribute("DlBandwidth",
                                     UintegerValue(bandwidthRbs));
    lteHelper->SetEnbDeviceAttribute("UlBandwidth",
                                     UintegerValue(bandwidthRbs));

    Ptr<Node> pgw = epcHelper->GetPgwNode();

    // Create remote host
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);
    InternetStackHelper internet;
    internet.Install(remoteHostContainer);

    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gb/s")));
    p2ph.SetDeviceAttribute("Mtu", UintegerValue(1500));
    p2ph.SetChannelAttribute("Delay", TimeValue(MilliSeconds(10)));
    NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);

    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(internetDevices);

    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"),
                                               Ipv4Mask("255.0.0.0"), 1);

    // Nodes
    NodeContainer enbNodes;
    enbNodes.Create(numCells);
    NodeContainer ueNodes;
    ueNodes.Create(numUes);

    // Mobility
    MobilityHelper enbMobility;
    Ptr<ListPositionAllocator> enbPositionAlloc =
        CreateObject<ListPositionAllocator>();
    for (uint32_t i = 0; i < numCells; ++i) {
      enbPositionAlloc->Add(Vector(i * 500.0, 0.0, 30.0));
    }
    enbMobility.SetPositionAllocator(enbPositionAlloc);
    enbMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    enbMobility.Install(enbNodes);

    MobilityHelper ueMobility;
    ueMobility.SetMobilityModel("ns3::ConstantVelocityMobilityModel");

    // Scenario: Commuter Rush (Clustered Start)
    if (congestionScenario == "commuter_rush") {
      ueMobility.SetPositionAllocator(
          "ns3::RandomRectanglePositionAllocator", "X",
          StringValue("ns3::UniformRandomVariable[Min=100|Max=200]"), "Y",
          StringValue("ns3::UniformRandomVariable[Min=90|Max=110]"));
    } else {
      // Default random distribution
      ueMobility.SetPositionAllocator(
          "ns3::RandomRectanglePositionAllocator", "X",
          StringValue("ns3::UniformRandomVariable[Min=0|Max=" +
                      std::to_string(numCells * 500) + "]"),
          "Y", StringValue("ns3::UniformRandomVariable[Min=-250|Max=250]"));
    }
    ueMobility.Install(ueNodes);

    // Set initial random velocities
    Ptr<UniformRandomVariable> xVar = CreateObject<UniformRandomVariable>();
    xVar->SetAttribute("Min", DoubleValue(0.0));
    xVar->SetAttribute("Max", DoubleValue(10.0));
    for (uint32_t i = 0; i < numUes; ++i) {
      if (congestionScenario == "commuter_rush") {
        // Start moving towards the handover boundary immediately
        ueNodes.Get(i)->GetObject<ConstantVelocityMobilityModel>()->SetVelocity(
            Vector(20.0, 0.0, 0.0));
      } else {
        ueNodes.Get(i)->GetObject<ConstantVelocityMobilityModel>()->SetVelocity(
            Vector(xVar->GetValue(), xVar->GetValue(), 0.0));
      }
    }

    // Devices
    NetDeviceContainer enbLteDevs = lteHelper->InstallEnbDevice(enbNodes);
    NetDeviceContainer ueLteDevs = lteHelper->InstallUeDevice(ueNodes);

    internet.Install(ueNodes);
    Ipv4InterfaceContainer ueIpIface =
        epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueLteDevs));

    for (uint32_t i = 0; i < numUes; ++i) {
      lteHelper->Attach(ueLteDevs.Get(i), enbLteDevs.Get(i % numCells));
    }

    // Enable X2 interface between all eNBs (required for handover)
    // Without this, the ANR table is empty and handover crashes with
    // "Cell ID X cannot be found in NRT"
    lteHelper->AddX2Interface(enbNodes);

    for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
      Ptr<Node> ueNode = ueNodes.Get(i);
      Ptr<Ipv4StaticRouting> ueStaticRouting =
          ipv4RoutingHelper.GetStaticRouting(ueNode->GetObject<Ipv4>());
      ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(),
                                       1);
    }

    // Apps (Simplified for Brevity)

    // Setup Application Traffic
    // Note: remoteHostContainer is needed but defined earlier as NodeContainer.
    ApplicationContainer trafficApps =
        SetupTraffic(ueNodes, remoteHostContainer, lteHelper);

    // Enable E2 interface
    Ptr<E2InterfaceManager> e2Manager;
    if (enableE2) {
      e2Manager = Create<E2InterfaceManager>(lteHelper, enbNodes, ueNodes,
                                             kafkaBrokers, topicSuffix);
      e2Manager->SetKpmInterval(MilliSeconds(kpmInterval));
      e2Manager->EnableE2();
    }

    // Congestion
    Ptr<CongestionScenarioManager> scenarioManager =
        Create<CongestionScenarioManager>(ueNodes, lteHelper, trafficApps);

    if (congestionScenario == "flash_crowd") {
      // Start congestion much earlier (0.2s) to see effects in short RL
      // episodes
      scenarioManager->ActivateScenario(CongestionScenarioManager::FLASH_CROWD,
                                        Seconds(0.2));
    } else if (congestionScenario == "mobility_storm") {
      scenarioManager->ActivateScenario(
          CongestionScenarioManager::MOBILITY_STORM, Seconds(0.2));
    } else if (congestionScenario == "traffic_burst") {
      scenarioManager->ActivateScenario(
          CongestionScenarioManager::TRAFFIC_BURST, Seconds(0.2));
    } else if (congestionScenario == "ping_pong") {
      scenarioManager->ActivateScenario(
          CongestionScenarioManager::HANDOVER_PING_PONG, Seconds(0.2));
    } else if (congestionScenario == "sleepy_campus") {
      scenarioManager->ActivateScenario(
          CongestionScenarioManager::SLEEPY_CAMPUS, Seconds(0.2));
    } else if (congestionScenario == "ambulance") {
      scenarioManager->ActivateScenario(CongestionScenarioManager::AMBULANCE,
                                        Seconds(5.0));
    } else if (congestionScenario == "adversarial") {
      scenarioManager->ActivateScenario(CongestionScenarioManager::ADVERSARIAL,
                                        Seconds(2.0));
    } else if (congestionScenario == "commuter_rush") {
      scenarioManager->ActivateScenario(
          CongestionScenarioManager::COMMUTER_RUSH, Seconds(0.2));
    } else if (congestionScenario == "mixed_reality") {
      scenarioManager->ActivateScenario(
          CongestionScenarioManager::MIXED_REALITY, Seconds(1.0));
    } else if (congestionScenario == "urban_canyon") {
      scenarioManager->ActivateScenario(CongestionScenarioManager::URBAN_CANYON,
                                        Seconds(1.0));
    } else if (congestionScenario == "iot_tsunami") {
      scenarioManager->ActivateScenario(CongestionScenarioManager::IOT_TSUNAMI,
                                        Seconds(0.5));
    } else if (congestionScenario == "spectrum_crunch") {
      scenarioManager->ActivateScenario(
          CongestionScenarioManager::SPECTRUM_CRUNCH, Seconds(0.5));
    }

    NS_LOG_INFO("Starting simulation...");
    // Ipv4GlobalRoutingHelper::PopulateRoutingTables(); // Caused crash with
    // LTE Optimize performance by disabling verbose logging
    // LogComponentEnable("UdpClient", LOG_LEVEL_INFO);
    // LogComponentEnable("UdpServer", LOG_LEVEL_INFO);
    LogComponentEnable("RadioCortexOranScenario", LOG_LEVEL_INFO);

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    NS_LOG_INFO("Simulation complete.");
    Simulator::Destroy();
  } catch (const std::exception &e) {
    std::cerr << "EXCEPTION CAUGHT: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "UNKNOWN EXCEPTION CAUGHT" << std::endl;
    return 1;
  }

  return 0;
}
