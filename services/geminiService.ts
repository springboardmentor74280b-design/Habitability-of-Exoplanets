
import { GoogleGenAI, Type } from "@google/genai";
import { PredictionResult } from "../types";

export async function predictHabitability(
  radius: number,
  mass: number,
  temp: number,
  period: number
): Promise<PredictionResult> {
  const apiKey = process.env.API_KEY || "";
  const ai = new GoogleGenAI({ apiKey });
  
  const prompt = `Assess the habitability of an exoplanet with the following parameters:
  - Radius: ${radius} Earth radii
  - Mass: ${mass} Earth masses
  - Equilibrium Temperature: ${temp} Kelvin
  - Orbital Period: ${period} days
  
  Consider factors like planetary density (derived from mass/radius), potential atmosphere based on gravity and temperature, and the likelihood of liquid water (assuming a 250K-373K range for liquid water).
  
  Provide a habitability score from 0.0 to 1.0 and a brief reasoning.`;

  try {
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            isHabitable: { type: Type.BOOLEAN },
            score: { type: Type.NUMBER, description: "Habitability score between 0 and 1" },
            reasoning: { type: Type.STRING }
          },
          required: ["isHabitable", "score", "reasoning"]
        }
      }
    });

    const result = JSON.parse(response.text || "{}");
    return {
      isHabitable: !!result.isHabitable,
      score: result.score || 0,
      reasoning: result.reasoning || "Analysis complete."
    };
  } catch (error) {
    console.error("Gemini Prediction Error:", error);
    // Fallback logic
    const score = Math.max(0, 1 - Math.abs(288 - temp) / 200 - Math.abs(1 - radius) / 5);
    return {
      isHabitable: score > 0.5,
      score: Number(score.toFixed(4)),
      reasoning: "Automatic heuristic analysis used due to API limitation."
    };
  }
}
