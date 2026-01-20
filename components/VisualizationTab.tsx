
import React from 'react';
import { 
  ScatterChart, 
  Scatter, 
  XAxis, 
  YAxis, 
  ZAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell
} from 'recharts';
import { TOP_PLANETS } from '../constants';

interface VisualizationTabProps {
  theme?: 'light' | 'dark';
}

const VisualizationTab: React.FC<VisualizationTabProps> = ({ theme = 'light' }) => {
  const isDark = theme === 'dark';
  const textColor = isDark ? '#64748b' : '#94a3b8';
  const gridColor = isDark ? '#1e293b' : '#f1f5f9';

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      if (data.name) {
        return (
          <div className={`p-5 rounded-2xl shadow-[0_10px_40px_rgba(0,0,0,0.3)] border-2 transition-all transform scale-100 animate-in zoom-in-95 ${
            isDark 
              ? 'bg-slate-900 border-slate-700 text-slate-200' 
              : 'bg-white border-slate-100 text-slate-800'
          }`}>
            <p className="font-black text-lg mb-3 border-b border-slate-100 dark:border-slate-800 pb-2 flex items-center gap-2 tracking-tight">
              <span className="text-blue-500">🪐</span> {data.name}
            </p>
            <div className="space-y-3">
              <div className="flex justify-between items-center gap-8">
                <span className="text-[10px] font-black uppercase tracking-widest opacity-60">Life Probability</span>
                <span className="font-mono text-blue-600 dark:text-blue-400 font-black text-lg">
                  {(data.habitabilityScore * 100).toFixed(4)}%
                </span>
              </div>
              <div className="grid grid-cols-2 gap-4 pt-1 border-t border-slate-100 dark:border-slate-800">
                <div className="flex flex-col">
                  <span className="text-[9px] uppercase opacity-50 font-black tracking-widest">Radius</span>
                  <span className="text-sm font-bold tabular-nums">{data.radius} R⊕</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-[9px] uppercase opacity-50 font-black tracking-widest">Temp</span>
                  <span className="text-sm font-bold tabular-nums">{data.temperature} K</span>
                </div>
              </div>
            </div>
          </div>
        );
      }
      if (data.range) {
        return (
          <div className={`p-4 rounded-xl shadow-2xl border ${
            isDark ? 'bg-slate-900 border-slate-700 text-slate-200' : 'bg-white border-slate-200 text-slate-800'
          }`}>
            <p className="text-[10px] font-black mb-1 opacity-50 uppercase tracking-widest">Range: {data.range} R⊕</p>
            <p className="text-base font-black">Count: {data.count} Worlds</p>
          </div>
        );
      }
    }
    return null;
  };

  const bins = [
    { range: '1.0-1.5', count: TOP_PLANETS.filter(p => p.radius >= 1.0 && p.radius < 1.5).length },
    { range: '1.5-2.0', count: TOP_PLANETS.filter(p => p.radius >= 1.5 && p.radius < 2.0).length },
    { range: '2.0-2.5', count: TOP_PLANETS.filter(p => p.radius >= 2.0 && p.radius <= 2.5).length },
  ];

  const getPointColor = (score: number) => {
    if (score >= 0.99) return '#10b981';
    if (score >= 0.98) return '#3b82f6';
    return '#f59e0b';
  };

  const ChartCard = ({ title, children, subtitle }: any) => (
    <div className="bg-white dark:bg-slate-900/40 backdrop-blur-md p-8 rounded-[2rem] shadow-xl border border-slate-100 dark:border-slate-800 transition-all hover:shadow-2xl hover:border-blue-500/20 group">
      <div className="mb-8">
        <h3 className="text-xl font-black text-slate-900 dark:text-white tracking-tight group-hover:text-blue-500 transition-colors">{title}</h3>
        <p className="text-xs font-bold text-slate-400 dark:text-slate-500 uppercase tracking-widest mt-1">{subtitle}</p>
      </div>
      <div className="h-[320px] w-full">
        {children}
      </div>
    </div>
  );

  return (
    <div className="max-w-7xl mx-auto py-12 px-6 space-y-12">
      <div className="flex flex-col items-center justify-center text-center mb-12">
        <h2 className="text-4xl md:text-5xl font-black text-slate-900 dark:text-white tracking-tighter mb-4">
          Data Viewport
        </h2>
        <p className="text-slate-500 dark:text-slate-400 max-w-2xl text-lg font-medium leading-relaxed">
          Interactive multi-dimensional analysis of habitability correlations across the exoplanet catalog.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 lg:gap-12">
        <ChartCard title="Habitability vs Radius" subtitle="Physical Scale Correlation">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 0, right: 0, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="4 4" vertical={false} stroke={gridColor} />
              <XAxis type="number" dataKey="radius" name="Radius" unit=" R⊕" domain={[1, 2.5]} stroke={textColor} fontSize={10} fontStyle="bold" />
              <YAxis type="number" dataKey="habitabilityScore" name="Score" domain={[0.97, 1.0]} stroke={textColor} fontSize={10} fontStyle="bold" />
              <ZAxis type="number" range={[150, 400]} />
              <Tooltip cursor={{ strokeDasharray: '4 4', stroke: '#3b82f6' }} content={<CustomTooltip />} />
              <Scatter name="Planets" data={TOP_PLANETS} fill="#3b82f6" />
            </ScatterChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Habitability vs Temperature" subtitle="Climate Energy Mapping">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 0, right: 0, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="4 4" vertical={false} stroke={gridColor} />
              <XAxis type="number" dataKey="temperature" name="Temp" unit=" K" domain={[290, 400]} stroke={textColor} fontSize={10} fontStyle="bold" />
              <YAxis type="number" dataKey="habitabilityScore" name="Score" domain={[0.97, 1.0]} stroke={textColor} fontSize={10} fontStyle="bold" />
              <ZAxis type="number" range={[150, 400]} />
              <Tooltip cursor={{ strokeDasharray: '4 4', stroke: '#0ea5e9' }} content={<CustomTooltip />} />
              <Scatter name="Planets" data={TOP_PLANETS} fill="#0ea5e9" />
            </ScatterChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Habitability vs Orbital Period" subtitle="Cyclical Stability Analysis">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 0, right: 0, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="4 4" vertical={false} stroke={gridColor} />
              <XAxis type="number" dataKey="orbitalPeriod" name="Period" unit=" d" domain={[0, 'auto']} stroke={textColor} fontSize={10} fontStyle="bold" />
              <YAxis type="number" dataKey="habitabilityScore" name="Score" domain={[0.97, 1.0]} stroke={textColor} fontSize={10} fontStyle="bold" />
              <ZAxis type="number" range={[150, 400]} />
              <Tooltip cursor={{ strokeDasharray: '4 4', stroke: '#6366f1' }} content={<CustomTooltip />} />
              <Scatter name="Planets" data={TOP_PLANETS}>
                {TOP_PLANETS.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getPointColor(entry.habitabilityScore)} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Dimension Distribution" subtitle="Planetary Scale Frequency">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={bins} margin={{ top: 0, right: 0, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="4 4" vertical={false} stroke={gridColor} />
              <XAxis dataKey="range" stroke={textColor} fontSize={10} fontStyle="bold" />
              <YAxis stroke={textColor} fontSize={10} fontStyle="bold" />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: isDark ? '#1e293b' : '#f8fafc' }} />
              <Bar dataKey="count" fill="#3b82f6" radius={[12, 12, 0, 0]}>
                {bins.map((entry, index) => (
                  <Cell key={`cell-${index}`} fillOpacity={0.8} fill={index === 2 ? '#1d4ed8' : '#3b82f6'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </div>
  );
};

export default VisualizationTab;
