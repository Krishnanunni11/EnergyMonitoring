'use client';
import React, { useState, useEffect, useCallback } from 'react';
import mqtt from 'mqtt';
import {
  Bell, Plug2, AlertCircle, PowerOff, Plus, Zap, Activity,
  Clock, LayoutGrid, BarChart3, Settings, Menu, Moon,
  ExternalLink, ChevronDown, TrendingUp, Sparkles, X, Save
} from 'lucide-react';

// --- TYPES ---

interface Appliance {
  id: string;
  name: string;
  voltage: string;
  current: string;
  history: number[][];
  prediction: string;
  score: number;
  predicted_power?: number;
  physics_power?: number;
  threshold?: number;
  deviation?: number;
  message?: string;
}

interface PredictionResult {
  plug_id?: string;
  appliance_id?: string;
  status?: string;
  avg_predicted_w?: number;
  avg_physics_w?: number;
  user_threshold_w?: number;
  deviation_w?: number;
  deviation_pct?: number;
  message?: string;
  usage_score?: number;
}

interface PredictionResponse {
  results?: PredictionResult[];
}

interface StatCardProps {
  icon: React.ComponentType<{ size?: number }>;
  label: string;
  value: string;
  unit: string;
  colorClass: string;
  bgClass: string;
}

interface DeviceCardProps {
  name: string;
  status: string;
  voltage: string;
  current: string;
  power: string;
  predictedPower?: number;
  threshold?: number;
  deviation?: number;
  message?: string;
  type?: 'online' | 'warning';
  score: number;
}

// --- UI COMPONENTS ---

const StatCard: React.FC<StatCardProps> = ({ icon: Icon, label, value, unit, colorClass, bgClass }) => (
  <div className="bg-white p-5 rounded-3xl shadow-sm border border-slate-100 flex items-center gap-4 flex-1 min-w-[200px]">
    <div className={`p-4 rounded-2xl ${bgClass} ${colorClass}`}>
      <Icon size={24} />
    </div>
    <div>
      <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">{label}</p>
      <div className="flex items-baseline gap-1">
        <span className="text-2xl font-bold text-slate-800">{value}</span>
        <span className="text-sm font-semibold text-slate-400 uppercase">{unit}</span>
      </div>
    </div>
  </div>
);

const DeviceCard: React.FC<DeviceCardProps> = ({ name, status, voltage, current, power, predictedPower, threshold, deviation, message, type = 'online', score }) => {
  const isWarning = type === 'warning';

  return (
    <div className={`p-6 rounded-[2.5rem] bg-white border shadow-sm transition-all duration-500 ${isWarning ? 'border-red-200 bg-red-50/30' : 'border-slate-50'
      }`}>
      <div className="flex justify-between items-start mb-6">
        <div className="flex items-center gap-4">
          <div className={`p-4 rounded-2xl ${isWarning ? 'bg-red-50 text-red-500' : 'bg-green-50 text-green-500'
            }`}>
            {isWarning ? <AlertCircle size={24} /> : <Plug2 size={24} />}
          </div>
          <div>
            <h4 className="font-bold text-slate-800">{name}</h4>
            <div className={`flex items-center gap-1.5 text-[10px] font-bold uppercase ${isWarning ? 'text-red-500' : 'text-green-500'
              }`}>
              <span className={`w-1.5 h-1.5 rounded-full ${isWarning ? 'bg-red-500 animate-pulse' : 'bg-green-500'}`} />
              {status}
            </div>
          </div>
        </div>
        {isWarning && (
          <div className="bg-red-500 text-white text-[10px] px-2 py-1 rounded-lg font-bold">
            RISK: {score}
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-y-4 mb-6">
        <div className="flex flex-col">
          <span className="text-[10px] text-slate-400 font-bold uppercase">Voltage</span>
          <div className="text-sm font-bold text-slate-700">{voltage} V</div>
        </div>
        <div className="flex flex-col">
          <span className="text-[10px] text-slate-400 font-bold uppercase">Current</span>
          <div className="text-sm font-bold text-slate-700">{current} A</div>
        </div>
        <div className="col-span-2 pt-2 border-t border-slate-50">
          <span className="text-[10px] text-slate-400 font-bold uppercase">Actual Power (V×I)</span>
          <div className="text-lg font-black text-blue-600">{power} W</div>
        </div>
        {predictedPower !== undefined && (
          <div className="col-span-2 pt-2">
            <span className="text-[10px] text-slate-400 font-bold uppercase">Predicted Power (AI)</span>
            <div className="text-lg font-black text-purple-600">{predictedPower.toFixed(2)} W</div>
          </div>
        )}
        {threshold !== undefined && (
          <div className="col-span-2 pt-2 border-t border-slate-100">
            <div className="flex justify-between items-center">
              <div>
                <span className="text-[10px] text-slate-400 font-bold uppercase">Threshold</span>
                <div className="text-sm font-bold text-slate-600">{threshold.toFixed(2)} W</div>
              </div>
              {deviation !== undefined && (
                <div className={`text-xs font-bold ${deviation > 0 ? 'text-red-600' : deviation < 0 ? 'text-orange-600' : 'text-green-600'}`}>
                  {deviation > 0 ? '+' : ''}{deviation.toFixed(2)} W
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {message ? (
        <p className={`text-[11px] font-bold p-2 rounded-lg ${isWarning ? 'bg-red-100/50 text-red-600' : 'bg-green-100/60 text-green-700'}`}>
          {isWarning ? '⚠️' : '✅'} {message}
        </p>
      ) : isWarning ? (
        <p className="text-[11px] text-red-600 font-bold bg-red-100/50 p-2 rounded-lg">
          ⚠️ LSTM: Abnormal Pattern Detected
        </p>
      ) : (
        <p className="text-[11px] text-green-700 font-bold bg-green-100/60 p-2 rounded-lg">
          ✅ LSTM: Normal Pattern Detected
        </p>
      )}
    </div>
  );
};

// --- THRESHOLD CONFIGURATION MODAL ---

interface ThresholdModalProps {
  isOpen: boolean;
  onClose: () => void;
  plugs: string[];
  onSave?: () => void;
}

interface ThresholdResponse {
  message?: string;
  device?: string;
  threshold?: number;
  error?: string;
}

type DeviceType =
  | 'air_conditioner'
  | 'bulb'
  | 'fan'
  | 'laptop'
  | 'microwave'
  | 'phone_charger'
  | 'refrigerator'
  | 'television'
  | 'washing_machine'
  | 'water_heater';

const ThresholdModal: React.FC<ThresholdModalProps> = ({ isOpen, onClose, plugs, onSave }) => {
  const [plugId, setPlugId] = useState<string>('');
  const [device, setDevice] = useState<DeviceType | ''>('');
  const [threshold, setThreshold] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [message, setMessage] = useState<string>('');

  const deviceTypes: DeviceType[] = [
    'air_conditioner', 'bulb', 'fan', 'laptop', 'microwave',
    'phone_charger', 'refrigerator', 'television', 'washing_machine', 'water_heater'
  ];

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    setMessage('');

    try {
      const res = await fetch('/api/set-threshold', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          plug_id: plugId,
          device: device,
          threshold: parseFloat(threshold)
        })
      });

      const data: ThresholdResponse = await res.json();

      if (res.ok) {
        setMessage(`✅ Threshold set: ${data.device} @ ${data.threshold}W`);
        onSave?.();
        setTimeout(() => {
          setPlugId('');
          setDevice('');
          setThreshold('');
          setMessage('');
        }, 2000);
      } else {
        setMessage(`❌ Error: ${data.error}`);
      }
    } catch (err) {
      setMessage('❌ Failed to connect to backend');
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-3xl shadow-2xl max-w-md w-full p-8 relative">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-2 hover:bg-slate-100 rounded-full transition"
        >
          <X size={20} />
        </button>

        <h2 className="text-2xl font-black text-slate-800 mb-2">Configure Threshold</h2>
        <p className="text-sm text-slate-500 mb-6">Set power limits for anomaly detection</p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-bold text-slate-600 uppercase mb-2">
              Plug ID
            </label>
            <select
              value={plugId}
              onChange={(e) => setPlugId(e.target.value)}
              className="w-full px-4 py-3 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            >
              <option value="">Select a plug...</option>
              {plugs.map((plug: string) => (
                <option key={plug} value={plug}>{plug}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-bold text-slate-600 uppercase mb-2">
              Device Type
            </label>
            <select
              value={device}
              onChange={(e) => setDevice(e.target.value as DeviceType)}
              className="w-full px-4 py-3 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            >
              <option value="">Select device type...</option>
              {deviceTypes.map((d) => (
                <option key={d} value={d}>{d.replace('_', ' ').toUpperCase()}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-bold text-slate-600 uppercase mb-2">
              Threshold (Watts)
            </label>
            <input
              type="number"
              step="0.01"
              value={threshold}
              onChange={(e) => setThreshold(e.target.value)}
              placeholder="e.g., 95.0"
              className="w-full px-4 py-3 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
          </div>

          {message && (
            <div className={`p-3 rounded-xl text-sm font-bold ${message.startsWith('✅') ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'
              }`}>
              {message}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-xl transition flex items-center justify-center gap-2 disabled:bg-slate-300"
          >
            {loading ? (
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              <>
                <Save size={18} />
                Save Threshold
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  );
};

// --- MAIN DASHBOARD ---

export default function Dashboard() {
  const [isSidebarOpen, setSidebarOpen] = useState<boolean>(false);
  const [mqttConnected, setMqttConnected] = useState<boolean>(false);
  const [isThresholdModalOpen, setIsThresholdModalOpen] = useState<boolean>(false);

  // 1. STATE: history stores [voltage, current] pairs for the LSTM
  const [appliances, setAppliances] = useState<Appliance[]>([]);

  // 2. MQTT WebSocket Connection
  useEffect(() => {
    const client = mqtt.connect("wss://broker.hivemq.com:8884/mqtt");

    client.on('connect', () => {
      console.log('✅ MQTT WebSocket Connected');
      setMqttConnected(true);
      // Subscribe to all smart plug topics
      client.subscribe('smart/plug/+/codedata', (err: Error | null) => {
        if (err) {
          console.error('MQTT Subscribe Error:', err);
        } else {
          console.log('📡 Subscribed to smart/plug/+/codedata');
        }
      });
    });

    client.on('message', (topic: string, message: Buffer) => {
      try {
        const data = JSON.parse(message.toString());
        const plugId = topic.split('/')[2]; // Extract plug ID from topic

        console.log('📥 MQTT Message:', plugId, data);

        const voltage = Number(data?.voltage ?? 0);
        const current = Number(data?.current ?? 0);

        setAppliances(prev => {
          const existing = prev.find(app => app.id === plugId);
          const newPair: number[] = [voltage, current];
          const updatedHistory = [...(existing?.history ?? []), newPair].slice(-10);

          if (existing) {
            return prev.map(app =>
              app.id === plugId
                ? {
                  ...app,
                  voltage: voltage.toFixed(1),
                  current: current.toFixed(2),
                  history: updatedHistory
                }
                : app
            );
          } else {
            return [...prev, {
              id: plugId,
              name: plugId,
              voltage: voltage.toFixed(1),
              current: current.toFixed(2),
              history: updatedHistory,
              prediction: 'Normal',
              score: 0
            }];
          }
        });
      } catch (err) {
        console.error('MQTT Message Parse Error:', err);
      }
    });

    client.on('error', (err: Error) => {
      console.error('MQTT Error:', err);
      setMqttConnected(false);
    });

    client.on('close', () => {
      console.log('❌ MQTT Disconnected');
      setMqttConnected(false);
    });

    return () => {
      client.end();
    };
  }, []);

  // 3. Fetch Latest Predictions from Backend
  useEffect(() => {
    const fetchPredictions = async () => {
      try {
        const res = await fetch('http://127.0.0.1:8000/latest-predictions', {
          cache: 'no-store'
        });

        if (!res.ok) return;

        const data = await res.json();

        if (Array.isArray(data.results)) {
          console.log('📊 Predictions received:', data.results);

          setAppliances(prev => prev.map(app => {
            const prediction = data.results.find((r: PredictionResult) => r.plug_id === app.id);

            if (prediction) {
              console.log(`✅ Updating ${app.id}:`, {
                predicted: prediction.avg_predicted_w,
                threshold: prediction.user_threshold_w,
                status: prediction.status
              });

              return {
                ...app,
                prediction: prediction.status === 'Normal' ? 'Normal' : 'Abnormal',
                predicted_power: prediction.avg_predicted_w,
                physics_power: prediction.avg_physics_w,
                threshold: prediction.user_threshold_w,
                deviation: prediction.deviation_w,
                message: prediction.message,
              };
            }
            return app;
          }));
        }
      } catch (err) {
        console.error('Fetch predictions error:', err);
      }
    };

    // Fetch immediately and then every 3 seconds
    fetchPredictions();
    const timer = setInterval(fetchPredictions, 3000);

    return () => clearInterval(timer);
  }, []);

  // 4. FALLBACK: HTTP Polling (commented out - using MQTT WebSocket as primary)
  /*
  useEffect(() => {
    const fetchSensorData = async () => {
      try {
        const res = await fetch('/api/sensor'); // Calls the Python /sensor-data endpoint
        const json = await res.json();

        if (Array.isArray(json.data)) {
          setAppliances(prev => {
            const previousById = new Map(prev.map(app => [app.id, app]));
            const latestSensorRowById = new Map<string, any>();

            for (const sensorRow of json.data) {
              const applianceId = typeof sensorRow?.appliance_id === 'string' ? sensorRow.appliance_id.trim() : '';
              if (!applianceId) continue;
              latestSensorRowById.set(applianceId, sensorRow);
            }

            return Array.from(latestSensorRowById.values())
              .map((sensorRow: any) => {
                const applianceId = sensorRow.appliance_id.trim();

                const previous = previousById.get(applianceId);
                const voltage = Number(sensorRow?.voltage ?? 0);
                const current = Number(sensorRow?.current ?? 0);

                const newPair: number[] = [voltage, current];
                const updatedHistory = [...(previous?.history ?? []), newPair].slice(-10);

                return {
                  id: applianceId,
                  name: applianceId,
                  voltage: voltage.toFixed(1),
                  current: current.toFixed(2),
                  history: updatedHistory,
                  prediction: typeof sensorRow?.status === 'string' ? sensorRow.status : previous?.prediction ?? 'Normal',
                  score: Number.isFinite(Number(sensorRow?.usage_score)) ? Number(sensorRow.usage_score) : previous?.score ?? 0
                };
              });
          });
        }
      } catch (err) {
        console.error("Sensor fetch error:", err);
      }
    };

    const timer = setInterval(fetchSensorData, 2000);
    return () => clearInterval(timer);
  }, []);
  */

  const totalLoad: number = appliances.reduce((sum, app) => sum + (parseFloat(app.voltage) * parseFloat(app.current)), 0) / 1000;
  const plugIds: string[] = appliances.map(app => app.id);

  return (
    <div className="min-h-screen bg-[#F8FAFC] flex font-sans">
      {/* Threshold Configuration Modal */}
      <ThresholdModal
        isOpen={isThresholdModalOpen}
        onClose={() => setIsThresholdModalOpen(false)}
        plugs={plugIds}
        onSave={() => {
          console.log('Threshold saved successfully');
        }}
      />

      {/* Sidebar */}
      <aside className={`fixed lg:sticky top-0 left-0 h-screen w-64 bg-white border-r p-6 z-50 transition-transform lg:translate-x-0 ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="flex items-center gap-3 mb-10">
          <div className="bg-blue-600 p-2 rounded-xl text-white"><Zap size={20} fill="currentColor" /></div>
          <span className="text-xl font-black text-blue-600 tracking-tighter">VoltFlow</span>
        </div>
        <nav className="space-y-1">
          <button className="w-full flex items-center gap-3 px-4 py-3 rounded-xl bg-blue-50 text-blue-600 font-bold"><LayoutGrid size={20} /> Panel</button>
          <button className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-slate-400 font-medium hover:bg-slate-50"><BarChart3 size={20} /> Usage</button>
          <button onClick={() => setIsThresholdModalOpen(true)} className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-slate-400 font-medium hover:bg-slate-50"><Settings size={20} /> Config</button>
        </nav>
      </aside>

      <main className="flex-1">
        <header className="p-6 bg-white border-b flex justify-between items-center sticky top-0 z-40">
          <div className="flex items-center gap-4">
            <button className="lg:hidden" onClick={() => setSidebarOpen(true)}><Menu /></button>
            <h1 className="font-black text-slate-800 uppercase tracking-widest text-sm">System Live Feed</h1>
          </div>
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border ${mqttConnected
            ? 'bg-green-50 text-green-600 border-green-100'
            : 'bg-red-50 text-red-600 border-red-100'
            }`}>
            <div className={`w-2 h-2 rounded-full ${mqttConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
              }`} />
            <span className="text-[10px] font-black uppercase">
              {mqttConnected ? 'MQTT Connected' : 'MQTT Disconnected'}
            </span>
          </div>
        </header>

        <div className="p-6 lg:p-10 space-y-10">
          <div className="flex flex-wrap gap-6">
            <StatCard icon={Zap} label="Total Load" value={totalLoad.toFixed(2)} unit="kW" colorClass="text-blue-500" bgClass="bg-blue-50" />
            <StatCard icon={Activity} label="Status" value={appliances.filter(a => a.prediction === 'Abnormal').length.toString()} unit="Alerts" colorClass="text-red-500" bgClass="bg-red-50" />
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-12 gap-10">
            {/* Grid of Devices */}
            <div className="xl:col-span-8 space-y-6">
              <h2 className="text-sm font-black text-slate-400 uppercase tracking-[0.2em]">Monitoring Nodes</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {appliances.map(app => (
                  <DeviceCard
                    key={app.id}
                    name={app.name}
                    voltage={app.voltage}
                    current={app.current}
                    power={(parseFloat(app.voltage) * parseFloat(app.current)).toFixed(1)}
                    predictedPower={app.predicted_power}
                    threshold={app.threshold}
                    deviation={app.deviation}
                    message={app.message}
                    status={app.prediction === 'Abnormal' ? 'Abnormal' : 'Normal'}
                    type={app.prediction === 'Abnormal' ? 'warning' : 'online'}
                    score={app.score}
                  />
                ))}
              </div>
            </div>

            {/* AI Status Sidebar */}
            <div className="xl:col-span-4">
              <div className="bg-slate-900 rounded-[3rem] p-8 text-white shadow-2xl relative overflow-hidden">
                <Sparkles className="absolute -right-6 -top-6 text-blue-500/20 w-32 h-32" />
                <h3 className="text-xl font-bold mb-4">PINN AI Analysis</h3>
                <p className="text-xs text-slate-400 leading-relaxed mb-6">
                  Automatic real-time anomaly detection using Physics-Informed Neural Network.
                </p>

                <div className="space-y-4">
                  <div className="bg-white/10 rounded-2xl p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold uppercase text-slate-400">Active Plugs</span>
                      <span className="text-2xl font-black text-white">{appliances.length}</span>
                    </div>
                  </div>

                  <div className="bg-white/10 rounded-2xl p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold uppercase text-slate-400">Anomalies</span>
                      <span className="text-2xl font-black text-red-400">{appliances.filter(a => a.prediction === 'Abnormal').length}</span>
                    </div>
                  </div>

                  <div className="bg-green-500/20 border border-green-500/30 rounded-2xl p-4">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                      <span className="text-xs font-bold text-green-300">AUTO-MONITORING ACTIVE</span>
                    </div>
                    <p className="text-[10px] text-green-200/60 mt-2">
                      Analysis triggers automatically after 20 MQTT readings per device
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}