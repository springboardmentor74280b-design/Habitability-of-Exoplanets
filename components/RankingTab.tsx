
import React, { useState, useMemo } from 'react';
import { TOP_PLANETS } from '../constants';
import { Exoplanet } from '../types';
import { jsPDF } from 'jspdf';

type SortKey = 'radius' | 'temperature' | 'habitabilityScore';
type SortOrder = 'asc' | 'desc';

const RankingTab: React.FC = () => {
  const [sortKey, setSortKey] = useState<SortKey>('habitabilityScore');
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc');
  const [selectedPlanet, setSelectedPlanet] = useState<Exoplanet | null>(null);
  const [isExporting, setIsExporting] = useState(false);

  const sortedPlanets = useMemo(() => {
    return [...TOP_PLANETS].sort((a, b) => {
      const aVal = a[sortKey];
      const bVal = b[sortKey];
      
      if (sortOrder === 'asc') {
        return aVal - bVal;
      } else {
        return bVal - aVal;
      }
    });
  }, [sortKey, sortOrder]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortOrder('desc');
    }
  };

  const exportToCSV = () => {
    setIsExporting(true);
    const headers = ['Rank', 'Name', 'Radius (Earth)', 'Mass (Earth)', 'Temp (K)', 'Period (Days)', 'Habitability Score'];
    const rows = sortedPlanets.map((p, i) => [
      i + 1,
      p.name,
      p.radius.toFixed(2),
      p.mass.toFixed(2),
      p.temperature.toFixed(0),
      p.orbitalPeriod.toFixed(1),
      (p.habitabilityScore * 100).toFixed(2) + '%'
    ]);

    const csvContent = [headers, ...rows].map(e => e.join(",")).join("\n");
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    link.setAttribute("download", `astrohabit_candidates_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(() => setIsExporting(false), 500);
  };

  const exportToPDF = () => {
    setIsExporting(true);
    const doc = new jsPDF();
    const timestamp = new Date().toLocaleString();

    // Technical Header
    doc.setFillColor(15, 23, 42); // slate-900
    doc.rect(0, 0, 210, 40, 'F');
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(22);
    doc.text('ASTROHABIT INTELLIGENCE REPORT', 15, 20);
    doc.setFontSize(10);
    doc.text(`Generated: ${timestamp} | Data: NASA Exoplanet Archive`, 15, 30);

    // Table Setup
    doc.setTextColor(50, 50, 50);
    doc.setFontSize(12);
    doc.text('Primary Candidate Leaderboard', 15, 55);
    
    let y = 65;
    const colWidths = [15, 50, 30, 30, 30, 35];
    const headers = ['#', 'Designation', 'Radius', 'Temp', 'Period', 'Score'];
    
    // Draw Table Headers
    doc.setFontSize(10);
    doc.setTextColor(100, 100, 100);
    let x = 15;
    headers.forEach((h, i) => {
      doc.text(h, x, y);
      x += colWidths[i];
    });

    y += 5;
    doc.setDrawColor(200, 200, 200);
    doc.line(15, y, 195, y);
    y += 10;

    // Draw Rows
    doc.setTextColor(0, 0, 0);
    sortedPlanets.forEach((p, idx) => {
      if (y > 270) {
        doc.addPage();
        y = 20;
      }
      let rowX = 15;
      doc.text((idx + 1).toString(), rowX, y); rowX += colWidths[0];
      doc.setFont('helvetica', 'bold');
      doc.text(p.name, rowX, y); rowX += colWidths[1];
      doc.setFont('helvetica', 'normal');
      doc.text(`${p.radius.toFixed(2)}`, rowX, y); rowX += colWidths[2];
      doc.text(`${p.temperature.toFixed(0)}K`, rowX, y); rowX += colWidths[3];
      doc.text(`${p.orbitalPeriod.toFixed(1)}d`, rowX, y); rowX += colWidths[4];
      doc.setTextColor(37, 99, 235); // blue-600
      doc.text(`${(p.habitabilityScore * 100).toFixed(1)}%`, rowX, y);
      doc.setTextColor(0, 0, 0);
      
      y += 8;
    });

    doc.save(`astrohabit_report_${new Date().toISOString().split('T')[0]}.pdf`);
    setTimeout(() => setIsExporting(false), 500);
  };

  const SortButton = ({ label, targetKey }: { label: string; targetKey: SortKey }) => (
    <button
      onClick={() => toggleSort(targetKey)}
      className={`px-5 py-2.5 rounded-xl text-xs font-black transition-all flex items-center gap-2 border tracking-widest uppercase ${
        sortKey === targetKey
          ? 'bg-blue-600 text-white border-blue-600 shadow-lg shadow-blue-500/30'
          : 'bg-white dark:bg-slate-800 text-slate-500 dark:text-slate-400 border-slate-200 dark:border-slate-700 hover:border-blue-400 hover:text-blue-500'
      }`}
    >
      {label}
      {sortKey === targetKey && (
        <span className="text-xs transition-transform duration-300">{sortOrder === 'asc' ? '↑' : '↓'}</span>
      )}
    </button>
  );

  return (
    <div className="max-w-7xl mx-auto py-12 px-6">
      <div className="flex flex-col items-center mb-12 text-center">
        <h2 className="text-4xl md:text-5xl font-black text-slate-900 dark:text-white tracking-tighter mb-4">
          Habitability Leaderboard
        </h2>
        <p className="text-slate-500 dark:text-slate-400 max-w-2xl text-lg font-medium">
          A definitive ranking of known worlds ordered by their AI-calculated potential for supporting life.
        </p>
      </div>

      <div className="flex flex-col lg:flex-row justify-between items-center gap-6 mb-10 pb-8 border-b border-slate-200 dark:border-slate-800">
        <div className="flex flex-wrap items-center justify-center gap-4">
          <span className="text-[10px] font-black text-slate-400 dark:text-slate-500 uppercase tracking-[0.2em] mr-2">Sort Criteria:</span>
          <SortButton label="Score" targetKey="habitabilityScore" />
          <SortButton label="Radius" targetKey="radius" />
          <SortButton label="Temp" targetKey="temperature" />
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={exportToCSV}
            disabled={isExporting}
            className="group flex items-center gap-2 px-5 py-2.5 bg-slate-100 dark:bg-slate-800 hover:bg-emerald-500 hover:text-white dark:hover:bg-emerald-600 text-slate-600 dark:text-slate-300 rounded-xl transition-all font-black text-xs uppercase tracking-widest border border-slate-200 dark:border-slate-700 disabled:opacity-50"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Excel / CSV
          </button>
          <button
            onClick={exportToPDF}
            disabled={isExporting}
            className="group flex items-center gap-2 px-5 py-2.5 bg-slate-900 dark:bg-white text-white dark:text-slate-900 hover:bg-blue-600 dark:hover:bg-blue-500 hover:text-white rounded-xl transition-all font-black text-xs uppercase tracking-widest shadow-xl disabled:opacity-50"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
            </svg>
            PDF Report
          </button>
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900/60 backdrop-blur-md rounded-3xl shadow-2xl overflow-hidden border border-slate-200 dark:border-slate-800">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-slate-100/50 dark:bg-slate-800/80 border-b border-slate-200 dark:border-slate-700">
                <th className="px-8 py-5 text-[10px] font-black text-slate-400 dark:text-slate-500 uppercase tracking-widest">Rank</th>
                <th className="px-8 py-5 text-[10px] font-black text-slate-400 dark:text-slate-500 uppercase tracking-widest">Designation</th>
                <th className="px-8 py-5 text-[10px] font-black text-slate-400 dark:text-slate-500 uppercase tracking-widest text-center">Dimension (R⊕)</th>
                <th className="px-8 py-5 text-[10px] font-black text-slate-400 dark:text-slate-500 uppercase tracking-widest text-center">Climate (K)</th>
                <th className="px-8 py-5 text-[10px] font-black text-slate-400 dark:text-slate-500 uppercase tracking-widest text-right">H-Index</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
              {sortedPlanets.map((planet, idx) => (
                <tr 
                  key={planet.id} 
                  onClick={() => setSelectedPlanet(planet)}
                  className="group hover:bg-blue-50/30 dark:hover:bg-blue-900/10 transition-colors cursor-pointer"
                >
                  <td className="px-8 py-6">
                    <span className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-400 dark:text-slate-500 text-xs font-black group-hover:bg-blue-100 dark:group-hover:bg-blue-900/40 group-hover:text-blue-600 transition-colors">
                      {idx + 1}
                    </span>
                  </td>
                  <td className="px-8 py-6">
                    <p className="font-black text-slate-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors text-lg tracking-tight">
                      {planet.name}
                    </p>
                    <p className="text-[10px] text-slate-400 dark:text-slate-500 font-bold uppercase tracking-widest">System Object</p>
                  </td>
                  <td className="px-8 py-6 text-center">
                    <span className="font-mono font-bold text-slate-600 dark:text-slate-300">{planet.radius.toFixed(2)}</span>
                  </td>
                  <td className="px-8 py-6 text-center">
                    <span className="font-mono font-bold text-slate-600 dark:text-slate-300">{planet.temperature.toFixed(0)}</span>
                  </td>
                  <td className="px-8 py-6 text-right">
                    <span className="text-xl font-black text-blue-600 dark:text-blue-400">
                      {(planet.habitabilityScore * 100).toFixed(1)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Detail Modal */}
      {selectedPlanet && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-6">
          <div 
            className="absolute inset-0 bg-slate-950/80 backdrop-blur-md transition-opacity duration-300" 
            onClick={() => setSelectedPlanet(null)}
          ></div>
          <div className="relative bg-white dark:bg-slate-900 rounded-[2.5rem] shadow-[0_0_100px_rgba(0,0,0,0.5)] w-full max-w-2xl overflow-hidden animate-in slide-in-from-bottom-8 zoom-in-95 duration-500 border border-white/20 dark:border-slate-800">
            
            {/* Modal Header */}
            <div className="bg-gradient-to-br from-slate-900 to-slate-950 p-10 text-white relative overflow-hidden">
              <div className="absolute top-0 right-0 p-8 opacity-10">
                <div className="w-32 h-32 rounded-full bg-white blur-3xl animate-pulse"></div>
              </div>
              <div className="relative flex justify-between items-start">
                <div>
                  <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/20 text-blue-400 text-[10px] font-black uppercase tracking-widest mb-4 border border-blue-500/20">
                    Object Profile
                  </div>
                  <h3 className="text-5xl font-black tracking-tighter mb-1">{selectedPlanet.name}</h3>
                  <p className="text-slate-400 font-bold tracking-widest text-xs uppercase">System Verification: 100%</p>
                </div>
                <button 
                  onClick={() => setSelectedPlanet(null)}
                  className="w-12 h-12 flex items-center justify-center rounded-2xl bg-white/5 hover:bg-white/10 transition-colors border border-white/10 group"
                >
                  <span className="text-xl group-hover:scale-125 transition-transform">✕</span>
                </button>
              </div>
            </div>
            
            {/* Modal Body */}
            <div className="p-10 space-y-10">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
                {[
                  { label: 'Mass', val: `${selectedPlanet.mass.toFixed(2)} M⊕` },
                  { label: 'Radius', val: `${selectedPlanet.radius.toFixed(2)} R⊕` },
                  { label: 'Climate', val: `${selectedPlanet.temperature.toFixed(0)} K` },
                  { label: 'Period', val: `${selectedPlanet.orbitalPeriod.toFixed(1)} D` }
                ].map((item, i) => (
                  <div key={i} className="space-y-1">
                    <p className="text-[10px] font-black text-slate-400 dark:text-slate-500 uppercase tracking-widest">{item.label}</p>
                    <p className="text-lg font-black text-slate-900 dark:text-white tabular-nums">{item.val}</p>
                  </div>
                ))}
              </div>

              <div className="p-8 bg-slate-50 dark:bg-slate-950/50 rounded-3xl border border-slate-100 dark:border-slate-800">
                <div className="flex items-center justify-between mb-4">
                   <p className="text-xs font-black text-slate-400 dark:text-slate-500 uppercase tracking-widest">Habitability Rating</p>
                   <span className="text-2xl font-black text-blue-600 dark:text-blue-400">{(selectedPlanet.habitabilityScore * 100).toFixed(2)}%</span>
                </div>
                <div className="w-full h-4 bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden shadow-inner">
                  <div 
                    className="h-full bg-gradient-to-r from-blue-600 to-cyan-500 rounded-full transition-all duration-1000 ease-out" 
                    style={{ width: `${selectedPlanet.habitabilityScore * 100}%` }}
                  ></div>
                </div>
              </div>

              <div className="flex flex-col md:flex-row gap-6">
                <div className="flex-grow p-6 bg-blue-50/50 dark:bg-blue-900/10 rounded-2xl border border-blue-100 dark:border-blue-900/30">
                  <p className="text-sm text-blue-800 dark:text-blue-300 leading-relaxed font-medium italic">
                    "Structural analysis suggests an Earth-like density with substantial atmosphere potential."
                  </p>
                </div>
                <button 
                  onClick={() => setSelectedPlanet(null)}
                  className="px-8 py-4 bg-slate-900 dark:bg-white text-white dark:text-slate-900 font-black rounded-2xl hover:scale-105 transition-transform shadow-xl whitespace-nowrap"
                >
                  Confirm Dismissal
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RankingTab;
