import React, { useState } from 'react';
import { TrainPage } from './pages/TrainPage';
import { ReplayPage } from './pages/ReplayPage';
import { ScanPage } from './pages/ScanPage';

type Lane = 'TRAIN' | 'REPLAY' | 'SCAN';

const App: React.FC = () => {
  const [currentLane, setCurrentLane] = useState<Lane>('SCAN');
  return (
    <div className="flex flex-col h-screen w-full bg-slate-900">
      {/* LANE SELECTOR - Top Navigation */}
      <div className="h-14 bg-slate-800 border-b border-slate-700 flex items-center px-4">
        <div className="flex space-x-2">
          {(['SCAN', 'REPLAY', 'TRAIN'] as Lane[]).map(lane => (
            <button
              key={lane}
              onClick={() => setCurrentLane(lane)}
              className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
                currentLane === lane
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
              }`}
            >
              {lane}
            </button>
          ))}
        </div>
        
        <div className="ml-auto flex items-center space-x-4">
          <div className="text-xs text-slate-400">
            <span className="text-slate-500">Mode:</span>{' '}
            <span className="font-medium">
              {currentLane === 'SCAN' && 'Strategy Discovery'}
              {currentLane === 'REPLAY' && 'Real-time Simulation'}
              {currentLane === 'TRAIN' && 'Training Data Analysis'}
            </span>
          </div>
        </div>
      </div>

      {/* LANE CONTENT */}
      <div className="flex-1 overflow-hidden">
        {currentLane === 'TRAIN' && <TrainPage />}
        {currentLane === 'REPLAY' && <ReplayPage />}
        {currentLane === 'SCAN' && <ScanPage />}
      </div>
    </div>
  );
};

export default App;