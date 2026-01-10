/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      /* 🔹 Semantic colors (REQUIRED) */
      colors: {
        border: "hsl(var(--border))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",

        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },

        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },

        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },

        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },

        /* 🔹 Your custom space theme (PRESERVED) */
        "space-blue": "#0B0D17",
        "neon-cyan": "#00F2FF",
        "galactic-purple": "#7000FF",
        "stellar-gold": "#FFD700",
      },

      /* 🔹 Fonts (PRESERVED) */
      fontFamily: {
        orbitron: ["Orbitron", "sans-serif"],
        inter: ["Inter", "sans-serif"],
      },

      /* 🔹 Animations (PRESERVED) */
      animation: {
        glow: "glow 2s ease-in-out infinite alternate",
        float: "float 6s ease-in-out infinite",
        "rotate-slow": "rotate 20s linear infinite",
      },

      /* 🔹 Keyframes (PRESERVED) */
      keyframes: {
        glow: {
          "0%": {
            boxShadow:
              "0 0 5px #00F2FF, 0 0 10px #00F2FF, 0 0 15px #00F2FF",
          },
          "100%": {
            boxShadow:
              "0 0 10px #00F2FF, 0 0 20px #00F2FF, 0 0 30px #00F2FF, 0 0 40px #7000FF",
          },
        },
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-20px)" },
        },
        rotate: {
          "0%": { transform: "rotate(0deg)" },
          "100%": { transform: "rotate(360deg)" },
        },
      },
    },
  },
  plugins: [],
};
