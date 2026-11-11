#include "ns3/core-module.h"
#include "ns3/applications-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/opengym-module.h"
//#include "ns3/csma-module.h"
#include "ns3/propagation-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/ipv4-flow-classifier.h"
#include "ns3/yans-wifi-channel.h"

#include <fstream>
#include <sstream>
#include <unistd.h>
#include <string>
#include <math.h>
#include <ctime>   //timestampi
#include <iomanip> // put_time
#include <deque>
#include <algorithm>
#include <csignal>
#include "scenario.h"

using namespace std;
using namespace ns3;

NS_LOG_COMPONENT_DEFINE("OpenGym");

void installTrafficGenerator(Ptr<ns3::Node> fromNode, Ptr<ns3::Node> toNode, int port, string offeredLoad, double startTime);
void PopulateARPcache();
void recordHistory();

double envStepTime = 0.01;
double simulationTime = 10; //seconds
double current_time = 0.0;
bool verbose = false;
int end_delay = 0;
bool dry_run = false;

Ptr<FlowMonitor> monitor;
FlowMonitorHelper flowmon;

void saveFlowMonitorSnapshot(int stationCount)
{
    if (monitor)
    {
        std::ostringstream xmlPath;
        xmlPath << "cw_edca_nWifi_" << stationCount
                << "_pid_" << getpid() << ".xml";
        monitor->SerializeToXmlFile(xmlPath.str(), true, true);
    }
}
ofstream outfile ("scratch/OSCAR_EDCA_DDPG/OSCAR_EDCA_DDPG_CW_data.csv", fstream::out);

uint32_t CW = 0;
uint32_t CWcomp = 0;


uint32_t history_length = 20;
deque<float> history;

string type = "discrete";
bool non_zero_start = false;
Scenario *wifiScenario;


/*
Define observation space
*/
Ptr<OpenGymSpace> MyGetObservationSpace(void)
{
    float low = 0.0;
    float high = 10.0;
    std::vector<uint32_t> shape = {
        history_length,
    };
    std::string dtype = TypeNameGet<float>();
    Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace>(low, high, shape, dtype);
    if (verbose)
        NS_LOG_UNCOND("MyGetObservationSpace: " << space);
    return space;
}

/*
Define action space
*/
Ptr<OpenGymSpace> MyGetActionSpace(void)
{
    float low = 0.0;
    float high = 896.0;
    std::vector<uint32_t> shape = {
        1,
    };
    std::string dtype = TypeNameGet<float>();
    Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace>(low, high, shape, dtype);
    if (verbose)
        NS_LOG_UNCOND("MyGetActionSpace: " << space);
    return space;
}

/*
Define extra info. Optional
*/
uint64_t g_rxPktNum = 0;
uint64_t g_txPktNum = 0;

double jain_index(void)
{
    double flowThr;
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
    std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();

    double nominator = 0.0;
    double denominator = 0.0;
    double n=0;
    double station_id = 0;
    for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin(); i != stats.end(); ++i)
    {
        flowThr = i->second.rxBytes;
        flowThr /= wifiScenario->getStationUptime(station_id, current_time);
        if(flowThr>0){
            nominator += flowThr;
            denominator += flowThr*flowThr;
            n++;
        }
        station_id++;
    }
    nominator *= nominator;
    denominator *= n;
    return nominator/denominator;
}

std::string MyGetExtraInfo(void)
{
    static float ticks = 0.0;
    static float lastValue = 0.0;
    float obs = g_rxPktNum - lastValue;
    lastValue = g_rxPktNum;
    ticks += envStepTime;

    float sentMbytes = obs * (1500 - 20 - 8 - 8) * 8.0 / 1024 / 1024;

    std::string myInfo = std::to_string(sentMbytes);
    myInfo = myInfo + "|" + to_string(CWcomp) + "|";
    myInfo = myInfo + to_string(wifiScenario->getActiveStationCount(ticks)) + "|";
    myInfo = myInfo + to_string(jain_index());

    if (verbose)
        NS_LOG_UNCOND("MyGetExtraInfo: " << myInfo);

    return myInfo;
}

/*
Execute received actions
*/

bool MyExecuteActions(Ptr<OpenGymDataContainer> action)
{
    if (verbose)
        NS_LOG_UNCOND("MyExecuteActions: " << action);

    Ptr<OpenGymBoxContainer<float>> box = DynamicCast<OpenGymBoxContainer<float>>(action);
    std::vector<float> actionVector = box->GetData();

    // OSCAR/DDPG's Python controller supplies a real action in [0, 896].
    // We interpret that action as the EDCA CW compensation value CWcomp.
    // This keeps the OSCAR continuous-control loop while changing what the
    // action controls: four EDCA access-category CW ranges instead of one BE CW.
    if (type == "discrete" || type == "continuous" || type == "direct_continuous")
    {
        float requested = actionVector.at(0);
        requested = std::max(0.0f, std::min(896.0f, requested));
        CWcomp = static_cast<uint32_t>(std::lround(requested));
    }
    else
    {
        std::cout << "Unsupported agent type!" << endl;
        exit(0);
    }

    CW = CWcomp;
    outfile << current_time << "," << CWcomp << endl;

    // EDCA parameters from Table 3 of the uploaded PDCF-DRL paper:
    // VO: [7, 15+CWcomp]
    // VI: [15+CWcomp, 31+CWcomp]
    // BE: [31+CWcomp, 1023]
    // BK: [31+CWcomp, 1023]
    uint32_t voMin = 7;
    uint32_t voMax = std::min(1023u, 15u + CWcomp);
    uint32_t viMin = std::min(1023u, 15u + CWcomp);
    uint32_t viMax = std::min(1023u, 31u + CWcomp);
    uint32_t beMin = std::min(1023u, 31u + CWcomp);
    uint32_t beMax = 1023;
    uint32_t bkMin = std::min(1023u, 31u + CWcomp);
    uint32_t bkMax = 1023;

    // Keep each range valid even after clamping.
    if (voMax < voMin) voMax = voMin;
    if (viMax < viMin) viMax = viMin;
    if (beMax < beMin) beMax = beMin;
    if (bkMax < bkMin) bkMax = bkMin;

    if(!dry_run){
        std::string base = "/$ns3::NodeListPriv/NodeList/*/$ns3::Node/DeviceList/*/$ns3::WifiNetDevice/Mac/$ns3::RegularWifiMac/";
        Config::Set(base + "VO_Txop/$ns3::QosTxop/MinCw", UintegerValue(voMin));
        Config::Set(base + "VO_Txop/$ns3::QosTxop/MaxCw", UintegerValue(voMax));
        Config::Set(base + "VI_Txop/$ns3::QosTxop/MinCw", UintegerValue(viMin));
        Config::Set(base + "VI_Txop/$ns3::QosTxop/MaxCw", UintegerValue(viMax));
        Config::Set(base + "BE_Txop/$ns3::QosTxop/MinCw", UintegerValue(beMin));
        Config::Set(base + "BE_Txop/$ns3::QosTxop/MaxCw", UintegerValue(beMax));
        Config::Set(base + "BK_Txop/$ns3::QosTxop/MinCw", UintegerValue(bkMin));
        Config::Set(base + "BK_Txop/$ns3::QosTxop/MaxCw", UintegerValue(bkMax));
    }
    return true;
}

float MyGetReward(void)
{
    static float ticks = 0.0;
    static uint32_t last_packets = 0;
    static float last_reward = 0.0;
    ticks += envStepTime;

    float res = g_rxPktNum - last_packets;
    float reward = res * (1500 - 20 - 8 - 8) * 8.0 / 1024 / 1024 / (5 * 150 * envStepTime) * 10;  // bits/s


    last_packets = g_rxPktNum;

    if (ticks <= 2 * envStepTime)
        return 0.0;

    if (verbose)
        NS_LOG_UNCOND("MyGetReward: " << reward);

    if(reward>1.0f || reward<0.0f)
        reward = last_reward;
    last_reward = reward;
    return last_reward;
}

/*
Collect observations
*/

Ptr<OpenGymDataContainer> MyGetObservation()
{
    recordHistory();

    std::vector<uint32_t> shape = {
        history_length,
    };
    Ptr<OpenGymBoxContainer<float>> box = CreateObject<OpenGymBoxContainer<float>>(shape);

    for (uint32_t i = 0; i < history.size(); i++)
    {
        if (history[i] >= -100 && history[i] <= 100)
            box->AddValue(history[i]);
        else
            box->AddValue(0);
    }
    for (uint32_t i = history.size(); i < history_length; i++)
    {
        box->AddValue(0);
    }
    if (verbose)
        NS_LOG_UNCOND("MyGetObservation: " << box);
    return box;
}

bool MyGetGameOver(void)
{

    return false;
}

void ScheduleNextStateRead(double envStepTime, Ptr<OpenGymInterface> openGymInterface)
{
    Simulator::Schedule(Seconds(envStepTime), &ScheduleNextStateRead, envStepTime, openGymInterface);
    openGymInterface->NotifyCurrentState();
}

void recordHistory()
{

    static uint32_t last_rx = 0;            // Previously received packets
    static uint32_t last_tx = 0;            // Previously transmitted packets
    static uint32_t calls = 0;              // Number of calls to this function
    calls++;
    current_time += envStepTime;

    float received = g_rxPktNum - last_rx;  // Received packets since the last observation
    float sent = g_txPktNum - last_tx;      // Sent (...)
    float errs = sent - received;           // Errors (...)
    float ratio;

    ratio = (sent > 0) ? (errs / sent) : 0.0f;
    history.push_front(ratio);

    // Remove the oldest observation if we have filled the history
    if (history.size() > history_length)
    {
        history.pop_back();
    }

    // Replace the last observation with the current one
    last_rx = g_rxPktNum;
    last_tx = g_txPktNum;

    if (calls < history_length && non_zero_start)
    {
        // Schedule the next observation if we are not at the end of the simulation
        Simulator::Schedule(Seconds(envStepTime), &recordHistory);
    }
    else if (calls == history_length && non_zero_start)
    {
        g_rxPktNum = 0;
        g_txPktNum = 0;
        last_rx = 0;
        last_tx = 0;
    }
}

void packetReceived(Ptr<const Packet> packet)
{
    NS_LOG_DEBUG("Client received a packet of " << packet->GetSize() << " bytes");
    g_rxPktNum++;
}

void packetSent(Ptr<const Packet> packet)
{
    g_txPktNum++;
}

void set_phy(int nWifi, int guardInterval, NodeContainer &wifiStaNode, NodeContainer &wifiApNode, YansWifiPhyHelper &phy)
{
    Ptr<MatrixPropagationLossModel> lossModel = CreateObject<MatrixPropagationLossModel>();
    lossModel->SetDefaultLoss(50);

    wifiStaNode.Create(nWifi);
    wifiApNode.Create(1);

    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    Ptr<YansWifiChannel> chan = channel.Create();
    chan->SetPropagationLossModel(lossModel);
    chan->SetPropagationDelayModel(CreateObject<ConstantSpeedPropagationDelayModel>());

    phy = YansWifiPhyHelper::Default();
    phy.SetChannel(chan);

    // Set guard interval
    phy.Set("GuardInterval", TimeValue(NanoSeconds(guardInterval)));
}

void set_nodes(int channelWidth, int rng, int32_t simSeed, NodeContainer wifiStaNode, NodeContainer wifiApNode, YansWifiPhyHelper phy, WifiMacHelper mac, WifiHelper wifi, NetDeviceContainer &apDevice)
{
    // Set the access point details
    Ssid ssid = Ssid("ns3-80211ax");

    mac.SetType("ns3::StaWifiMac",
                "Ssid", SsidValue(ssid),
                "ActiveProbing", BooleanValue(false),
                "BE_MaxAmpduSize", UintegerValue(0),
                "QosSupported", BooleanValue(true),
                "VO_MaxAmpduSize", UintegerValue(0),
                "VI_MaxAmpduSize", UintegerValue(0),
                "BK_MaxAmpduSize", UintegerValue(0));

    NetDeviceContainer staDevice;
    staDevice = wifi.Install(phy, mac, wifiStaNode);

    mac.SetType("ns3::ApWifiMac",
                "EnableBeaconJitter", BooleanValue(false),
                "Ssid", SsidValue(ssid),
                "QosSupported", BooleanValue(true),
                "VO_MaxAmpduSize", UintegerValue(0),
                "VI_MaxAmpduSize", UintegerValue(0),
                "BE_MaxAmpduSize", UintegerValue(0),
                "BK_MaxAmpduSize", UintegerValue(0));

    apDevice = wifi.Install(phy, mac, wifiApNode);

    // Set channel width
    Config::Set("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/ChannelWidth", UintegerValue(channelWidth));

    // mobility.
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();

    positionAlloc->Add(Vector(0.0, 0.0, 0.0));
    positionAlloc->Add(Vector(1.0, 0.0, 0.0));
    mobility.SetPositionAllocator(positionAlloc);

    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");

    mobility.Install(wifiApNode);
    mobility.Install(wifiStaNode);
    /* Internet stack*/
    InternetStackHelper stack;
    stack.Install(wifiApNode);
    stack.Install(wifiStaNode);

    //Random
    if(simSeed!=-1)
        RngSeedManager::SetSeed(simSeed);
    RngSeedManager::SetRun(rng);

    Ipv4AddressHelper address;
    address.SetBase("192.168.1.0", "255.255.255.0");
    Ipv4InterfaceContainer staNodeInterface;
    Ipv4InterfaceContainer apNodeInterface;

    staNodeInterface = address.Assign(staDevice);
    apNodeInterface = address.Assign(apDevice);

    // EDCA default parameters from the uploaded paper (Table 2).
    // AIFSN: VO=2, VI=2, BE=3, BK=7. TXOP is set to the paper values.
    std::string base = "/$ns3::NodeListPriv/NodeList/*/$ns3::Node/DeviceList/*/$ns3::WifiNetDevice/Mac/$ns3::RegularWifiMac/";
    Config::Set(base + "VO_Txop/$ns3::QosTxop/Aifsn", UintegerValue(2));
    Config::Set(base + "VI_Txop/$ns3::QosTxop/Aifsn", UintegerValue(2));
    Config::Set(base + "BE_Txop/$ns3::QosTxop/Aifsn", UintegerValue(3));
    Config::Set(base + "BK_Txop/$ns3::QosTxop/Aifsn", UintegerValue(7));
    Config::Set(base + "VO_Txop/$ns3::QosTxop/TxopLimit", TimeValue(MicroSeconds(1504)));
    Config::Set(base + "VI_Txop/$ns3::QosTxop/TxopLimit", TimeValue(MicroSeconds(3008)));
    Config::Set(base + "BE_Txop/$ns3::QosTxop/TxopLimit", TimeValue(MicroSeconds(0)));
    Config::Set(base + "BK_Txop/$ns3::QosTxop/TxopLimit", TimeValue(MicroSeconds(0)));

    // Initial EDCA CW values. The first RL action will update these during the run.
    Config::Set(base + "VO_Txop/$ns3::QosTxop/MinCw", UintegerValue(7));
    Config::Set(base + "VO_Txop/$ns3::QosTxop/MaxCw", UintegerValue(15));
    Config::Set(base + "VI_Txop/$ns3::QosTxop/MinCw", UintegerValue(15));
    Config::Set(base + "VI_Txop/$ns3::QosTxop/MaxCw", UintegerValue(31));
    Config::Set(base + "BE_Txop/$ns3::QosTxop/MinCw", UintegerValue(31));
    Config::Set(base + "BE_Txop/$ns3::QosTxop/MaxCw", UintegerValue(1023));
    Config::Set(base + "BK_Txop/$ns3::QosTxop/MinCw", UintegerValue(31));
    Config::Set(base + "BK_Txop/$ns3::QosTxop/MaxCw", UintegerValue(1023));
}

void set_sim(bool tracing, bool dry_run, int warmup, uint32_t openGymPort, YansWifiPhyHelper phy, NetDeviceContainer apDevice, int end_delay, int stationCount, Ptr<FlowMonitor> &monitor, FlowMonitorHelper &flowmon)
{
    monitor = flowmon.InstallAll();
    monitor->SetAttribute("StartTime", TimeValue(Seconds(warmup)));

    if (simulationTime > 0.1)
    {
        Simulator::Schedule(
            Seconds(simulationTime - 0.1),
            &saveFlowMonitorSnapshot,
            stationCount
        );
    }

    if (tracing)
    {
        phy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
        phy.EnablePcap("cw", apDevice.Get(0));
    }

    Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface>(openGymPort);
    openGymInterface->SetGetActionSpaceCb(MakeCallback(&MyGetActionSpace));
    openGymInterface->SetGetObservationSpaceCb(MakeCallback(&MyGetObservationSpace));
    openGymInterface->SetGetGameOverCb(MakeCallback(&MyGetGameOver));
    openGymInterface->SetGetObservationCb(MakeCallback(&MyGetObservation));
    openGymInterface->SetGetRewardCb(MakeCallback(&MyGetReward));
    openGymInterface->SetGetExtraInfoCb(MakeCallback(&MyGetExtraInfo));
    openGymInterface->SetExecuteActionsCb(MakeCallback(&MyExecuteActions));


    if (non_zero_start)
    {
        Simulator::Schedule(Seconds(1.0), &recordHistory);
        Simulator::Schedule(Seconds(envStepTime * history_length + 1.0), &ScheduleNextStateRead, envStepTime, openGymInterface);
    }
    else
        Simulator::Schedule(Seconds(1.0), &ScheduleNextStateRead, envStepTime, openGymInterface);


    Simulator::Stop(Seconds(simulationTime + end_delay + 1.0 + envStepTime*(history_length+1)));

    NS_LOG_UNCOND("Simulation started");
    Simulator::Run();
}

void signalHandler(int)
{
    Simulator::Stop();
}

int main(int argc, char *argv[])
{
    int nWifi = 5;
    bool tracing = false;
    bool useRts = false;
    int mcs = 11;
    int channelWidth = 20;
    int guardInterval = 800;
    string offeredLoad = "150";
    int port = 1025;
    string outputCsv = "cw_edca_5.csv";
    string scenario = "basic";
    dry_run = false;

    int rng = 1;
    int warmup = 1;

    uint32_t openGymPort = 5555;
    int32_t simSeed = -1;

    signal(SIGTERM, signalHandler);
    outfile << "SimulationTime,CWcomp" << endl;

    CommandLine cmd;
    cmd.AddValue("openGymPort", "Specify port number. Default: 5555", openGymPort);
    cmd.AddValue("CW", "Value of Contention Window", CW);
    cmd.AddValue("historyLength", "Length of history window", history_length);
    cmd.AddValue("nWifi", "Number of wifi 802.11ax STA devices", nWifi);
    cmd.AddValue("verbose", "Tell echo applications to log if true", verbose);
    cmd.AddValue("tracing", "Enable pcap tracing", tracing);
    cmd.AddValue("rng", "Number of RngRun", rng);
    cmd.AddValue("simTime", "Simulation time in seconds. Default: 10s", simulationTime);
    cmd.AddValue("envStepTime", "Step time in seconds. Default: 0.1s", envStepTime);
    cmd.AddValue("agentType", "Type of agent actions: discrete, continuous", type);
    cmd.AddValue("nonZeroStart", "Start only after history buffer is filled", non_zero_start);
    cmd.AddValue("scenario", "Scenario for analysis: basic, convergence, reaction", scenario);
    cmd.AddValue("dryRun", "Execute scenario with BEB and no agent interaction", dry_run);
    cmd.AddValue("seed", "Random seed", simSeed);

    cmd.Parse(argc, argv);


    NS_LOG_UNCOND("Ns3Env parameters:");
    NS_LOG_UNCOND("--nWifi: " << nWifi);
    NS_LOG_UNCOND("--simulationTime: " << simulationTime);
    NS_LOG_UNCOND("--openGymPort: " << openGymPort);
    NS_LOG_UNCOND("--envStepTime: " << envStepTime);
    NS_LOG_UNCOND("--seed: " << simSeed);
    NS_LOG_UNCOND("--agentType: " << type);
    NS_LOG_UNCOND("--scenario: " << scenario);
    NS_LOG_UNCOND("--dryRun: " << dry_run);

    if (verbose)
    {
        LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
        LogComponentEnable("UdpEchoServerApplication", LOG_LEVEL_INFO);
    }

    if (useRts)
    {
        Config::SetDefault("ns3::WifiRemoteStationManager::RtsCtsThreshold", StringValue("0"));
    }

    NodeContainer wifiStaNode;
    NodeContainer wifiApNode;
    YansWifiPhyHelper phy;
    set_phy(nWifi, guardInterval, wifiStaNode, wifiApNode, phy);

    WifiMacHelper mac;
    WifiHelper wifi;

    wifi.SetStandard(WIFI_PHY_STANDARD_80211ax_5GHZ);

    std::ostringstream oss;
    oss << "HeMcs" << mcs;
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager", "DataMode", StringValue(oss.str()),
                                 "ControlMode", StringValue(oss.str()));


    NetDeviceContainer apDevice;
    set_nodes(channelWidth, rng, simSeed, wifiStaNode, wifiApNode, phy, mac, wifi, apDevice);

    ScenarioFactory helper = ScenarioFactory(nWifi, wifiStaNode, wifiApNode, port, offeredLoad, history_length);
    wifiScenario = helper.getScenario(scenario);


    if (non_zero_start)
        end_delay = envStepTime * history_length + 1.0;
    else
        end_delay = 0.0;


    wifiScenario->installScenario(simulationTime + end_delay + envStepTime, envStepTime, MakeCallback(&packetReceived));

    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxBegin", MakeCallback(&packetSent));

    wifiScenario->PopulateARPcache();
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();


    set_sim(tracing, dry_run, warmup, openGymPort, phy, apDevice, end_delay, nWifi, monitor, flowmon);
    {
        std::ofstream diag("/tmp/edca_writer_diag_20261101.txt", std::ios::app);
        diag << "Returned from set_sim\\n";
    }
    NS_LOG_UNCOND("DIAGNOSTIC: returned from set_sim; entering CSV writer.");

    double flowThr;
    float res =  g_rxPktNum * (1500 - 20 - 8 - 8) * 8.0 / 1024 / 1024;
    printf("Sent mbytes: %.2f\tThroughput: %.3f", res, res/simulationTime);
    ofstream myfile;
    myfile.open(outputCsv, ios::app);
    {
        std::ofstream diag("/tmp/edca_writer_diag_20261101.txt", std::ios::app);
        diag << "CSV open=" << myfile.is_open() << " path=" << outputCsv << "\\n";
    }
    if (myfile.is_open()) {
        NS_LOG_UNCOND("DIAGNOSTIC: CSV opened: " << outputCsv);
    } else {
        NS_LOG_UNCOND("DIAGNOSTIC: CSV could not be opened: " << outputCsv);
    }


    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
    std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();
    for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin(); i != stats.end(); ++i)
    {
        auto time = std::time(nullptr); 
        auto tm = *std::localtime(&time);
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(i->first);
        flowThr = i->second.rxBytes * 8.0 / simulationTime / 1000 / 1000;
        NS_LOG_UNCOND("Flow " << i->first << " (" << t.sourceAddress << " -> " << t.destinationAddress << ")\tThroughput: " << flowThr << " Mbps\tTime: " << i->second.timeLastRxPacket.GetSeconds() - i->second.timeFirstTxPacket.GetSeconds() << " s\tRx packets " << i->second.rxPackets);
        myfile << std::put_time(&tm, "%Y-%m-%d %H:%M") << "," << CWcomp << "," << nWifi << "," << RngSeedManager::GetRun() << "," << t.sourceAddress << "," << t.destinationAddress << "," << flowThr;
        myfile << std::endl;
    }
    myfile.close();

    Simulator::Destroy();
    NS_LOG_UNCOND("Packets registered by handler: " << g_rxPktNum << " Packets; final CWcomp=" << CWcomp << endl);

    return 0;
}
