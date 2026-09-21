import { Routes, Route, Link, NavLink } from "react-router-dom";
import HomePage from "./pages/HomePage";
import GamePage from "./pages/GamePage";
import AnalysisPage from "./pages/AnalysisPage";
import AboutPage from "./pages/AboutPage";

function Nav() {
  const base = "px-3 py-2 rounded text-sm font-medium transition-colors";
  const active = `${base} bg-blue-600 text-white`;
  const inactive = `${base} text-slate-700 hover:bg-slate-200`;
  return (
    <nav className="bg-white shadow-sm border-b border-slate-200">
      <div className="max-w-6xl mx-auto px-4 flex items-center gap-2 h-14">
        <Link to="/" className="font-bold text-blue-700 mr-4">PCSO Lotto Stats</Link>
        <NavLink to="/" end className={({ isActive }) => isActive ? active : inactive}>Home</NavLink>
        <NavLink to="/analysis" className={({ isActive }) => isActive ? active : inactive}>Analysis</NavLink>
        <NavLink to="/about" className={({ isActive }) => isActive ? active : inactive}>About</NavLink>
      </div>
    </nav>
  );
}

export default function App() {
  return (
    <div className="min-h-screen">
      <Nav />
      <main className="max-w-6xl mx-auto px-4 py-8">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/game/:id" element={<GamePage />} />
          <Route path="/analysis" element={<AnalysisPage />} />
          <Route path="/about" element={<AboutPage />} />
        </Routes>
      </main>
    </div>
  );
}
