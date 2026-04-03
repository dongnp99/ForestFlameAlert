/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        fire: {
          low:    '#1D9E75',
          medium: '#EF9F27',
          high:   '#D85A30',
          extreme:'#E24B4A',
        },
      },
    },
  },
  plugins: [],
};
