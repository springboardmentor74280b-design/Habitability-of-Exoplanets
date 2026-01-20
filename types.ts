
export interface Exoplanet {
  id: string;
  name: string;
  radius: number; // in Earth radii
  mass: number; // in Earth masses
  temperature: number; // in Kelvin
  orbitalPeriod: number; // in days
  habitabilityScore: number; // 0.0 to 1.0
}

export interface PredictionResult {
  isHabitable: boolean;
  score: number;
  reasoning: string;
}

export enum TabType {
  PREDICTION = 'prediction',
  RANKING = 'ranking',
  VISUALIZATION = 'visualization'
}
