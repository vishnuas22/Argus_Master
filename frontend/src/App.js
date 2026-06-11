import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import { Crosshair } from "@phosphor-icons/react";
import Dashboard from "./pages/Dashboard";
import History from "./pages/History";
import VerdictPage from "./pages/VerdictPage";
import "./App.css";

const navClass = ({ isActive }) =>
  `px-3 py-1.5 text-xs font-mono uppercase tracking-[0.2em] transition-colors duration-200 ${
    isActive ? "text-zinc-50 border-b border-zinc-50" : "text-zinc-500 hover:text-zinc-300"
  }`;

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-zinc-950 text-zinc-50">
        <header className="border-b border-zinc-800 sticky top-0 z-40 bg-zinc-950/90 backdrop-blur-md">
          <div className="max-w-[1600px] mx-auto px-4 md:px-8 h-14 flex items-center justify-between">
            <NavLink to="/" className="flex items-center gap-3" data-testid="argus-logo">
              <Crosshair size={20} weight="light" className="text-emerald-500" />
              <div className="flex items-baseline gap-3">
                <span className="text-base font-semibold tracking-[0.3em]">ARGUS</span>
                <span className="hidden sm:inline text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-500">
                  image evidence court
                </span>
              </div>
            </NavLink>
            <nav className="flex items-center gap-2">
              <NavLink to="/" end className={navClass} data-testid="nav-assess">
                Assess
              </NavLink>
              <NavLink to="/history" className={navClass} data-testid="nav-history">
                History
              </NavLink>
            </nav>
          </div>
        </header>
        <main className="max-w-[1600px] mx-auto p-4 md:p-6 lg:p-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/history" element={<History />} />
            <Route path="/verdict/:id" element={<VerdictPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
