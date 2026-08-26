/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      colors: {
        primary: {
          DEFAULT: '#111827',
        },
        secondary: {
          DEFAULT: '#6b7280',
        },
        accent: {
          DEFAULT: '#16a34a',
        }
      }
    },
  },
  plugins: [],
}
