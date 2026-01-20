
import React, { useState, useMemo } from 'react';
import { predictHabitability } from '../services/geminiService';
import { PredictionResult } from '../types';

const BOUNDS = {
  radius: { min: 0.1, max: 20, label: 'Radius' },
  mass: { min: 0.1, max: 50, label: 'Mass' },
  temp: { min: 50, max: 4000, label: 'Temperature' },
  period: { min: 0.1, max: 100000, label: 'Period' }
};

const PredictionTab: React.FC = () => {
  const [params, setParams] = useState({
    radius: 1.0,
    mass: 1.0,
    temp: 288,
    period: 365
  });
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);

  // Validation logic
  const errors = useMemo(() => {
    const errs: Record<string, string> = {};
    if (params.radius < BOUNDS.radius.min || params.radius > BOUNDS.radius.max) {
      errs.radius = `Must be between ${BOUNDS.radius.min} and ${BOUNDS.radius.max}`;
    }
    if (params.mass < BOUNDS.mass.min || params.mass > BOUNDS.mass.max) {
      errs.mass = `Must be between ${BOUNDS.mass.min} and ${BOUNDS.mass.max}`;
    }
    if (params.temp < BOUNDS.temp.min || params.temp > BOUNDS.temp.max) {
      errs.temp = `Must be between ${BOUNDS.temp.min} and ${BOUNDS.temp.max}`;
    }
    if (params.period < BOUNDS.period.min || params.period > BOUNDS.period.max) {
      errs.period = `Must be between ${BOUNDS.period.min} and ${BOUNDS.period.max}`;
    }
    return errs;
  }, [params]);

  const hasErrors = Object.keys(errors).length > 0;

  const handlePredict = async () => {
    if (hasErrors) return;
    setLoading(true);
    try {
      const res = await predictHabitability(params.radius, params.mass, params.temp, params.period);
      setResult(res);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const InputField = ({ label, id, value, unit, onChange, step = "1", error }: any) => (
    <div className="group relative">
      <label className={`block text-[10px] font-black uppercase tracking-widest mb-2 transition-colors ${error ? 'text-red-500' : 'text-slate-400 dark:text-slate-500 group-focus-within:text-blue-500'}`}>
        {label}
      </label>
      <div className="relative">
        <input
          type="number"
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
          className={`w-full pl-4 pr-12 py-3 bg-slate-50 dark:bg-slate-800/50 border rounded-xl focus:ring-4 outline-none text-slate-900 dark:text-white font-bold transition-all text-lg shadow-inner ${
            error 
              ? 'border-red-500 focus:ring-red-500/10 focus:border-red-500' 
              : 'border-slate-200 dark:border-slate-700 focus:ring-blue-500/10 focus:border-blue-500'
          }`}
          step={step}
        />
        <span className={`absolute right-4 top-1/2 -translate-y-1/2 text-xs font-bold pointer-events-none transition-colors ${error ? 'text-red-400' : 'text-slate-400 dark:text-slate-600'}`}>
          {unit}
        </span>
      </div>
      {error && (
        <p className="absolute -bottom-5 left-0 text-[9px] font-bold text-red-500 uppercase tracking-tighter">
          {error}
        </p>
      )}
    </div>
  );

  return (
    <div className="max-w-5xl mx-auto py-12 px-6">
      <div className="flex flex-col items-center text-center mb-12">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 text-[10px] font-black uppercase tracking-widest mb-4">
          <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse"></span>
          AI Prediction Engine
        </div>
        <h2 className="text-4xl md:text-5xl font-black text-slate-900 dark:text-white tracking-tighter mb-4">
          Habitability Discovery
        </h2>
        <p className="text-slate-500 dark:text-slate-400 max-w-xl text-lg font-medium leading-relaxed">
          Input astronomical coordinates and planetary data to evaluate the life-bearing potential of distant worlds.
        </p>
      </div>

      <div className="bg-white dark:bg-slate-900/40 backdrop-blur-xl rounded-3xl shadow-2xl p-8 md:p-12 mb-10 border border-white/20 dark:border-slate-800">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-12">
          <InputField 
            label="Planetary Radius" 
            value={params.radius} 
            unit="R⊕" 
            step="0.01" 
            error={errors.radius}
            onChange={(val: number) => setParams({ ...params, radius: val })} 
          />
          <InputField 
            label="Planetary Mass" 
            value={params.mass} 
            unit="M⊕" 
            step="0.01" 
            error={errors.mass}
            onChange={(val: number) => setParams({ ...params, mass: val })} 
          />
          <InputField 
            label="Surface Temp" 
            value={params.temp} 
            unit="K" 
            error={errors.temp}
            onChange={(val: number) => setParams({ ...params, temp: val })} 
          />
          <InputField 
            label="Orbital Cycle" 
            value={params.period} 
            unit="Days" 
            error={errors.period}
            onChange={(val: number) => setParams({ ...params, period: val })} 
          />
        </div>

        <button
          onClick={handlePredict}
          disabled={loading || hasErrors}
          className={`w-full group relative overflow-hidden py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-black text-lg rounded-2xl shadow-xl transition-all hover:-translate-y-0.5 active:translate-y-0 ${
            loading || hasErrors ? 'opacity-50 cursor-not-allowed shadow-none grayscale' : 'shadow-blue-500/20 hover:shadow-blue-500/40'
          }`}
        >
          <div className="relative z-10 flex items-center justify-center gap-3">
            {loading ? (
              <>
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                <span>Analyzing Astronomical Context...</span>
              </>
            ) : (
              <>
                <span>{hasErrors ? 'Invalid Input Parameters' : 'Execute Deep Analysis'}</span>
                {!hasErrors && (
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                  </svg>
                )}
              </>
            )}
          </div>
          {!loading && !hasErrors && (
            <div className="absolute inset-0 bg-white/10 translate-y-full group-hover:translate-y-0 transition-transform duration-500"></div>
          )}
        </button>
      </div>

      {result && (
        <div className={`rounded-3xl border-2 p-10 shadow-2xl transition-all animate-in zoom-in-95 duration-700 relative overflow-hidden ${
          result.isHabitable 
            ? 'bg-emerald-50/30 dark:bg-emerald-900/10 border-emerald-500/20 text-emerald-950 dark:text-emerald-100 shadow-emerald-500/10' 
            : 'bg-amber-50/30 dark:bg-amber-900/10 border-amber-500/20 text-amber-950 dark:text-amber-100 shadow-amber-500/10'
        }`}>
          <div className={`absolute top-0 left-0 bottom-0 w-2 ${result.isHabitable ? 'bg-emerald-500' : 'bg-amber-500'}`}></div>
          
          <div className="relative flex flex-col items-center text-center gap-6">
            <div className={`w-24 h-24 rounded-full flex items-center justify-center text-4xl shadow-inner ${result.isHabitable ? 'bg-emerald-100 dark:bg-emerald-800/30 text-emerald-600' : 'bg-amber-100 dark:bg-amber-800/30 text-amber-600'}`}>
              {result.isHabitable ? '🌱' : '🌋'}
            </div>
            
            <div>
              <p className="text-xs font-black uppercase tracking-[0.3em] opacity-50 mb-2">Final Evaluation</p>
              <h3 className="text-3xl md:text-4xl font-black mb-2">
                {result.isHabitable ? 'Habitable Candidate' : 'Non-Habitable Profile'}
              </h3>
              <div className={`text-5xl font-black tabular-nums ${result.isHabitable ? 'text-emerald-500' : 'text-amber-500'}`}>
                {(result.score * 100).toFixed(1)}%
              </div>
            </div>

            <div className="max-w-3xl bg-white/40 dark:bg-black/20 p-6 rounded-2xl border border-white/40 dark:border-white/5 text-lg font-medium leading-relaxed italic">
              "{result.reasoning}"
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionTab;
