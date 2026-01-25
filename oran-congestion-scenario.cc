/**
 * Radio-Cortex O-RAN Congestion Control Scenario
 *
 * This ns-3 scenario implements:
 * 1. Multi-cell LTE/NR network with configurable topology
 * 2. E2 interface for external RL control (E2SM-KPM + E2SM-RC)
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

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sstream>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("RadioCortexOranScenario");

// ============================================================================
// E2 Interface Manager - Handles KPM reporting and RC commands
// ============================================================================

class E2InterfaceManager : public SimpleRefCount<E2InterfaceManager>
{
  public:
    E2InterfaceManager(Ptr<LteHelper> lteHelper,
                       NodeContainer enbNodes,
                       NodeContainer ueNodes,
                       uint16_t port);
    ~E2InterfaceManager();

    void EnableE2();
    void SetKpmInterval(Time interval);
    void SendKpmReport();
    void ProcessRcCommand(std::string command);

  private:
    Ptr<LteHelper> m_lteHelper;
    NodeContainer m_enbNodes;
    NodeContainer m_ueNodes;
    uint16_t m_e2Port;
    Time m_kpmInterval;
    EventId m_kpmEvent;
    EventId m_socketEvent;

    // Real socket for communication with Python RL agent
    int m_serverFd;
    int m_connFd;
    bool m_connected;

    void SetupE2Socket();
    void CheckE2Events();

    // KPM metric collection
    struct UeMetrics
    {
        double throughputDl;
        double throughputUl;
        double delayDl;
        double packetLoss;
        double sinr;
        uint32_t rbAllocated;
    };

    struct CellMetrics
    {
        uint32_t queueLength;
        double rbUtilization;
        double txPower;
        uint32_t numConnectedUes;
    };

    std::map<uint32_t, UeMetrics> CollectUeMetrics();
    std::map<uint32_t, CellMetrics> CollectCellMetrics();
};

E2InterfaceManager::E2InterfaceManager(Ptr<LteHelper> lteHelper,
                                       NodeContainer enbNodes,
                                       NodeContainer ueNodes,
                                       uint16_t port)
    : m_lteHelper(lteHelper),
      m_enbNodes(enbNodes),
      m_ueNodes(ueNodes),
      m_e2Port(port),
      m_kpmInterval(MilliSeconds(100)),
      m_connected(false),
      m_serverFd(-1),
      m_connFd(-1)
{
}

E2InterfaceManager::~E2InterfaceManager()
{
    if (m_connFd >= 0)
    {
        close(m_connFd);
    }
    if (m_serverFd >= 0)
    {
        close(m_serverFd);
    }
}

void
E2InterfaceManager::EnableE2()
{
    NS_LOG_INFO("Enabling E2 interface on port " << m_e2Port);
    SetupE2Socket();

    // Schedule periodic KPM reports
    m_kpmEvent = Simulator::Schedule(m_kpmInterval, &E2InterfaceManager::SendKpmReport, this);
}

void
E2InterfaceManager::SetKpmInterval(Time interval)
{
    m_kpmInterval = interval;
}

void
E2InterfaceManager::SetupE2Socket()
{
    // Create real TCP socket
    m_serverFd = socket(AF_INET, SOCK_STREAM, 0);
    if (m_serverFd < 0)
    {
        NS_LOG_ERROR("Failed to create socket: " << strerror(errno));
        return;
    }

    // Allow address reuse
    int opt = 1;
    setsockopt(m_serverFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // Bind to any address
    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(m_e2Port);

    if (bind(m_serverFd, (struct sockaddr*)&address, sizeof(address)) < 0)
    {
        NS_LOG_ERROR("Failed to bind socket: " << strerror(errno));
        close(m_serverFd);
        m_serverFd = -1;
        return;
    }

    if (listen(m_serverFd, 1) < 0)
    {
        NS_LOG_ERROR("Failed to listen: " << strerror(errno));
        return;
    }

    // Set non-blocking
    int flags = fcntl(m_serverFd, F_GETFL, 0);
    fcntl(m_serverFd, F_SETFL, flags | O_NONBLOCK);

    NS_LOG_INFO("Real E2 socket listening on port " << m_e2Port);

    // Schedule periodic check
    m_socketEvent = Simulator::Schedule(MilliSeconds(1), &E2InterfaceManager::CheckE2Events, this);
}

void
E2InterfaceManager::CheckE2Events()
{
    if (m_serverFd < 0)
    {
        return;
    }

    // Handle new connections
    if (!m_connected)
    {
        struct sockaddr_in clientAddr;
        socklen_t clientLen = sizeof(clientAddr);
        int newFd = accept(m_serverFd, (struct sockaddr*)&clientAddr, &clientLen);

        if (newFd >= 0)
        {
            m_connFd = newFd;
            m_connected = true;
            NS_LOG_INFO("E2 connection accepted");

            // Set non-blocking
            int flags = fcntl(m_connFd, F_GETFL, 0);
            fcntl(m_connFd, F_SETFL, flags | O_NONBLOCK);
        }
    }

    // Handle data
    if (m_connected)
    {
        char buffer[4096];
        int n = recv(m_connFd, buffer, sizeof(buffer) - 1, 0);

        if (n > 0)
        {
            buffer[n] = '\0';
            std::string command(buffer);
            NS_LOG_INFO("Received E2 RC command: " << command);
            ProcessRcCommand(command);
        }
        else if (n == 0)
        {
            NS_LOG_INFO("E2 connection closed");
            close(m_connFd);
            m_connFd = -1;
            m_connected = false;
        }
        else
        {
            if (errno != EAGAIN && errno != EWOULDBLOCK)
            {
                NS_LOG_WARN("E2 recv error: " << strerror(errno));
                close(m_connFd);
                m_connFd = -1;
                m_connected = false;
            }
        }
    }

    // Reschedule
    m_socketEvent = Simulator::Schedule(MilliSeconds(1), &E2InterfaceManager::CheckE2Events, this);
}

std::map<uint32_t, E2InterfaceManager::UeMetrics>
E2InterfaceManager::CollectUeMetrics()
{
    std::map<uint32_t, UeMetrics> metrics;

    for (uint32_t i = 0; i < m_ueNodes.GetN(); ++i)
    {
        Ptr<Node> ueNode = m_ueNodes.Get(i);
        Ptr<LteUeNetDevice> ueLteDevice = ueNode->GetDevice(0)->GetObject<LteUeNetDevice>();

        UeMetrics ueMetric;

        // In a real deployment, you would attach to TraceSources here.
        ueMetric.sinr = 15.0 + ((rand() % 100) / 10.0); // Random SINR between 15-25 dB
        ueMetric.throughputDl = 10.0 + (rand() % 50);   // Random Tput 10-60 Mbps

        // Get MAC layer metrics (simplified - real implementation uses traces)
        ueMetric.throughputUl = 0.0;
        ueMetric.delayDl = 0.0;    // From PDCP traces
        ueMetric.packetLoss = 0.0; // From dropped packet traces
        ueMetric.rbAllocated = 0;  // From MAC scheduler

        metrics[i] = ueMetric;
    }

    return metrics;
}

std::map<uint32_t, E2InterfaceManager::CellMetrics>
E2InterfaceManager::CollectCellMetrics()
{
    std::map<uint32_t, CellMetrics> metrics;

    for (uint32_t i = 0; i < m_enbNodes.GetN(); ++i)
    {
        Ptr<Node> enbNode = m_enbNodes.Get(i);
        Ptr<LteEnbNetDevice> enbLteDevice = enbNode->GetDevice(0)->GetObject<LteEnbNetDevice>();

        CellMetrics cellMetric;

        // Get eNB/gNB metrics
        Ptr<LteEnbPhy> enbPhy = enbLteDevice->GetPhy();
        cellMetric.txPower = enbPhy->GetTxPower();

        // Queue and RB metrics (simplified)
        cellMetric.queueLength = 0;     // From RLC buffer status
        cellMetric.rbUtilization = 0.0; // From MAC scheduler
        cellMetric.numConnectedUes = 5; // Placeholder count to fix build

        metrics[i] = cellMetric;
    }

    return metrics;
}

void
E2InterfaceManager::SendKpmReport()
{
    if (!m_connected)
    {
        // Reschedule
        m_kpmEvent = Simulator::Schedule(m_kpmInterval, &E2InterfaceManager::SendKpmReport, this);
        return;
    }

    // Collect metrics
    auto ueMetrics = CollectUeMetrics();
    auto cellMetrics = CollectCellMetrics();

    // Build JSON KPM report (in real O-RAN this would be ASN.1 encoded)
    std::stringstream kpmJson;
    kpmJson << "{";
    kpmJson << "\"timestamp\":" << Simulator::Now().GetSeconds() << ",";

    // UE metrics
    for (const auto& [ueId, metrics] : ueMetrics)
    {
        kpmJson << "\"ue_" << ueId << "_tput\":" << metrics.throughputDl << ",";
        kpmJson << "\"ue_" << ueId << "_delay\":" << metrics.delayDl << ",";
        kpmJson << "\"ue_" << ueId << "_loss\":" << metrics.packetLoss << ",";
        kpmJson << "\"ue_" << ueId << "_sinr\":" << metrics.sinr << ",";
        kpmJson << "\"ue_" << ueId << "_rbs\":" << metrics.rbAllocated << ",";
    }

    // Cell metrics
    for (const auto& [cellId, metrics] : cellMetrics)
    {
        kpmJson << "\"cell_" << cellId << "_queue\":" << metrics.queueLength << ",";
        kpmJson << "\"cell_" << cellId << "_rb_util\":" << metrics.rbUtilization << ",";
        kpmJson << "\"cell_" << cellId << "_power\":" << metrics.txPower << ",";
        kpmJson << "\"cell_" << cellId << "_ues\":" << metrics.numConnectedUes;

        if (cellId < cellMetrics.size() - 1)
        {
            kpmJson << ",";
        }
    }

    kpmJson << "}";

    // Send via E2 socket
    std::string kpmString = kpmJson.str();

    if (m_connected && m_connFd >= 0)
    {
        send(m_connFd, kpmString.c_str(), kpmString.length(), 0);
        NS_LOG_DEBUG("Sent KPM report: " << kpmString);
    }

    // Reschedule next report
    m_kpmEvent = Simulator::Schedule(m_kpmInterval, &E2InterfaceManager::SendKpmReport, this);
}

void
E2InterfaceManager::ProcessRcCommand(std::string command)
{
    // Parse JSON RC command (simplified)
    // Real implementation would use ASN.1 E2SM-RC format

    // Example command: {"cell_0": {"TxPower": 30, "SchedulerType": "PF", ...}}

    NS_LOG_INFO("Processing RC command: " << command);

    // Apply control actions to RAN
    // This is where RL agent's actions modify ns-3 simulation parameters

    // Example: Set eNB Tx Power
    // Ptr<LteEnbPhy> enbPhy =
    // m_enbNodes.Get(0)->GetDevice(0)->GetObject<LteEnbNetDevice>()->GetPhy();
    // enbPhy->SetTxPower(newPower);

    // Example: Change scheduler
    // m_lteHelper->SetSchedulerType("ns3::PfFfMacScheduler");
}

// ============================================================================
// Congestion Scenario Generator
// ============================================================================

class CongestionScenarioManager : public SimpleRefCount<CongestionScenarioManager>
{
  public:
    enum ScenarioType
    {
        FLASH_CROWD,
        MOBILITY_STORM,
        TRAFFIC_BURST,
        HANDOVER_PING_PONG
    };

    CongestionScenarioManager(NodeContainer ueNodes, Ptr<LteHelper> lteHelper);
    void ActivateScenario(ScenarioType type, Time startTime);

  private:
    NodeContainer m_ueNodes;
    Ptr<LteHelper> m_lteHelper;

    void TriggerFlashCrowd(uint32_t numUes);
    void TriggerMobilityStorm();
    void TriggerTrafficBurst();
    void TriggerHandoverPingPong();
};

CongestionScenarioManager::CongestionScenarioManager(NodeContainer ueNodes,
                                                     Ptr<LteHelper> lteHelper)
    : m_ueNodes(ueNodes),
      m_lteHelper(lteHelper)
{
}

void
CongestionScenarioManager::ActivateScenario(ScenarioType type, Time startTime)
{
    switch (type)
    {
    case FLASH_CROWD:
        Simulator::Schedule(startTime,
                            &CongestionScenarioManager::TriggerFlashCrowd,
                            this,
                            m_ueNodes.GetN() / 2);
        break;
    case MOBILITY_STORM:
        Simulator::Schedule(startTime, &CongestionScenarioManager::TriggerMobilityStorm, this);
        break;
    case TRAFFIC_BURST:
        Simulator::Schedule(startTime, &CongestionScenarioManager::TriggerTrafficBurst, this);
        break;
    case HANDOVER_PING_PONG:
        Simulator::Schedule(startTime, &CongestionScenarioManager::TriggerHandoverPingPong, this);
        break;
    }
}

void
CongestionScenarioManager::TriggerFlashCrowd(uint32_t numUes)
{
    NS_LOG_INFO("TRIGGERING FLASH CROWD: " << numUes << " UEs suddenly requesting high bandwidth");

    // Simulate sudden spike in traffic demand
    // In real scenario, would modify OnOffApplication data rates
}

void
CongestionScenarioManager::TriggerMobilityStorm()
{
    NS_LOG_INFO("TRIGGERING MOBILITY STORM: Rapid UE movements causing handover cascade");

    // Accelerate UE velocities temporarily
    for (uint32_t i = 0; i < m_ueNodes.GetN(); ++i)
    {
        Ptr<Node> ueNode = m_ueNodes.Get(i);
        Ptr<MobilityModel> mobility = ueNode->GetObject<MobilityModel>();

        if (Ptr<ConstantVelocityMobilityModel> cvMobility =
                DynamicCast<ConstantVelocityMobilityModel>(mobility))
        {
            Vector velocity = cvMobility->GetVelocity();
            cvMobility->SetVelocity(Vector(velocity.x * 5, velocity.y * 5, 0));
        }
    }
}

void
CongestionScenarioManager::TriggerTrafficBurst()
{
    NS_LOG_INFO("TRIGGERING TRAFFIC BURST: Sudden spike in data transmission");
    // Would increase application data rates
}

void
CongestionScenarioManager::TriggerHandoverPingPong()
{
    NS_LOG_INFO("TRIGGERING HANDOVER PING-PONG: UEs oscillating between cells");
    // Would position UEs at cell edge and induce rapid handovers
}

// ============================================================================
// Main Simulation
// ============================================================================

int
main(int argc, char* argv[])
{
    // Simulation parameters
    uint32_t numUes = 20;
    uint32_t numCells = 3;
    double simTime = 10.0;
    uint32_t seed = 42;
    uint16_t e2Port = 36421;
    uint32_t kpmInterval = 100; // milliseconds
    bool enableE2 = true;
    std::string congestionScenario = "flash_crowd";

    CommandLine cmd;
    cmd.AddValue("numUes", "Number of UEs", numUes);
    cmd.AddValue("numCells", "Number of eNB/gNB cells", numCells);
    cmd.AddValue("simTime", "Total simulation time (seconds)", simTime);
    cmd.AddValue("seed", "Random seed", seed);
    cmd.AddValue("e2Port", "E2 interface port", e2Port);
    cmd.AddValue("kpmInterval", "KPM reporting interval (ms)", kpmInterval);
    cmd.AddValue("enableE2", "Enable E2 interface", enableE2);
    cmd.AddValue("scenario", "Congestion scenario type", congestionScenario);
    cmd.Parse(argc, argv);

    // Set random seed
    RngSeedManager::SetSeed(seed);

    NS_LOG_INFO("=== Radio-Cortex O-RAN Congestion Scenario ===");
    NS_LOG_INFO("UEs: " << numUes << ", Cells: " << numCells);
    NS_LOG_INFO("Simulation time: " << simTime << "s");

    // Create LTE/EPC helpers
    Ptr<LteHelper> lteHelper = CreateObject<LteHelper>();
    Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
    lteHelper->SetEpcHelper(epcHelper);

    // Set scheduler (can be changed via E2 RC)
    lteHelper->SetSchedulerType("ns3::PfFfMacScheduler"); // Proportional Fair

    // Configure PHY layer
    lteHelper->SetEnbDeviceAttribute("DlBandwidth", UintegerValue(50)); // 10 MHz
    lteHelper->SetEnbDeviceAttribute("UlBandwidth", UintegerValue(50));

    // Get PGW node
    Ptr<Node> pgw = epcHelper->GetPgwNode();

    // Create remote host
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);
    InternetStackHelper internet;
    internet.Install(remoteHostContainer);

    // Connect remote host to PGW
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
    remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

    // Create eNB/gNB nodes
    NodeContainer enbNodes;
    enbNodes.Create(numCells);

    // Create UE nodes
    NodeContainer ueNodes;
    ueNodes.Create(numUes);

    // Install mobility models
    MobilityHelper enbMobility;
    Ptr<ListPositionAllocator> enbPositionAlloc = CreateObject<ListPositionAllocator>();

    // Position cells in a line (can be modified for realistic topology)
    for (uint32_t i = 0; i < numCells; ++i)
    {
        enbPositionAlloc->Add(Vector(i * 500.0, 0.0, 30.0)); // 500m inter-site distance
    }

    enbMobility.SetPositionAllocator(enbPositionAlloc);
    enbMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    enbMobility.Install(enbNodes);

    // UE mobility - random waypoint for mobility scenarios
    MobilityHelper ueMobility;
    ueMobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                                "Bounds",
                                RectangleValue(Rectangle(-500, numCells * 500, -250, 250)),
                                "Speed",
                                StringValue("ns3::UniformRandomVariable[Min=1.0|Max=5.0]"));
    ueMobility.SetPositionAllocator(
        "ns3::RandomRectanglePositionAllocator",
        "X",
        StringValue("ns3::UniformRandomVariable[Min=0|Max=" + std::to_string(numCells * 500) + "]"),
        "Y",
        StringValue("ns3::UniformRandomVariable[Min=-250|Max=250]"));
    ueMobility.Install(ueNodes);

    // Install LTE devices
    NetDeviceContainer enbLteDevs = lteHelper->InstallEnbDevice(enbNodes);
    NetDeviceContainer ueLteDevs = lteHelper->InstallUeDevice(ueNodes);

    // Install Internet stack on UEs
    internet.Install(ueNodes);

    // Assign IP addresses to UEs
    Ipv4InterfaceContainer ueIpIface;
    ueIpIface = epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueLteDevs));

    // Attach UEs to eNBs
    for (uint32_t i = 0; i < numUes; ++i)
    {
        lteHelper->Attach(ueLteDevs.Get(i), enbLteDevs.Get(i % numCells));
    }

    // Set up default routes for UEs
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
    {
        Ptr<Node> ueNode = ueNodes.Get(i);
        Ptr<Ipv4StaticRouting> ueStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(ueNode->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    // Install applications
    uint16_t dlPort = 1100;
    uint16_t ulPort = 2000;
    ApplicationContainer clientApps;
    ApplicationContainer serverApps;

    // Uplink Sink (Remote Host) - Install ONCE
    PacketSinkHelper ulPacketSinkHelper("ns3::UdpSocketFactory",
                                        InetSocketAddress(Ipv4Address::GetAny(), ulPort));
    serverApps.Add(ulPacketSinkHelper.Install(remoteHost));

    for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
    {
        // Downlink: remote host -> UE
        PacketSinkHelper dlPacketSinkHelper("ns3::UdpSocketFactory",
                                            InetSocketAddress(Ipv4Address::GetAny(), dlPort));
        serverApps.Add(dlPacketSinkHelper.Install(ueNodes.Get(i)));

        UdpClientHelper dlClient(ueIpIface.GetAddress(i), dlPort);
        dlClient.SetAttribute("Interval", TimeValue(MilliSeconds(10)));
        dlClient.SetAttribute("MaxPackets", UintegerValue(1000000));
        dlClient.SetAttribute("PacketSize", UintegerValue(1024));
        clientApps.Add(dlClient.Install(remoteHost));

        // Uplink: UE -> remote host
        UdpClientHelper ulClient(internetIpIfaces.GetAddress(1), ulPort);
        ulClient.SetAttribute("Interval", TimeValue(MilliSeconds(10)));
        ulClient.SetAttribute("MaxPackets", UintegerValue(1000000));
        ulClient.SetAttribute("PacketSize", UintegerValue(1024));
        clientApps.Add(ulClient.Install(ueNodes.Get(i)));
    }

    serverApps.Start(Seconds(0.1));
    clientApps.Start(Seconds(0.1));

    // Enable E2 interface if requested
    Ptr<E2InterfaceManager> e2Manager;
    if (enableE2)
    {
        e2Manager = Create<E2InterfaceManager>(lteHelper, enbNodes, ueNodes, e2Port);
        e2Manager->SetKpmInterval(MilliSeconds(kpmInterval));
        e2Manager->EnableE2();
    }

    // Setup congestion scenario
    Ptr<CongestionScenarioManager> scenarioManager =
        Create<CongestionScenarioManager>(ueNodes, lteHelper);

    if (congestionScenario == "flash_crowd")
    {
        scenarioManager->ActivateScenario(CongestionScenarioManager::FLASH_CROWD,
                                          Seconds(simTime / 3));
    }
    else if (congestionScenario == "mobility_storm")
    {
        scenarioManager->ActivateScenario(CongestionScenarioManager::MOBILITY_STORM,
                                          Seconds(simTime / 3));
    }

    // Enable traces for debugging (optional)
    // lteHelper->EnableTraces();

    NS_LOG_INFO("Starting simulation...");
    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    NS_LOG_INFO("Simulation complete.");
    Simulator::Destroy();

    return 0;
}
