import React, { useState, useEffect } from 'react';
import { Camera, AlertTriangle, Clock, User } from 'lucide-react';

const PPEAlarmInterface = () => {
  const [currentTime, setCurrentTime] = useState(new Date());
  
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Sample data for demonstration
  const detectionHistory = [
    { type: 'No Helmet', count: 2 },
    { type: 'No Mask', count: 3 },
    { type: 'No Safety Vest', count: 1 },
    { type: 'No Gloves', count: 1 }
  ];

  const totalPeople = 15;
  const totalMissingPPE = detectionHistory.reduce((sum, item) => sum + item.count, 0);
  const compliancePercentage = Math.round(((totalPeople - totalMissingPPE) / totalPeople) * 100);

  // Severity color calculation based on compliance percentage
  const getSeverityColor = (percentage) => {
    if (percentage >= 90) return { color: 'bg-green-500', text: 'LOW', textColor: 'text-green-800' };
    if (percentage >= 75) return { color: 'bg-yellow-500', text: 'MEDIUM', textColor: 'text-yellow-800' };
    if (percentage >= 60) return { color: 'bg-orange-500', text: 'HIGH', textColor: 'text-orange-800' };
    return { color: 'bg-red-500', text: 'CRITICAL', textColor: 'text-red-800' };
  };

  const severity = getSeverityColor(compliancePercentage);

  // Sample violation screenshots
  const violationScreenshots = [
    { id: 1, person: 'Person A', violation: 'No Helmet', timestamp: '14:23:15', image: '/api/placeholder/120/160' },
    { id: 2, person: 'Person B', violation: 'No Mask', timestamp: '14:22:45', image: '/api/placeholder/120/160' },
    { id: 3, person: 'Person C', violation: 'No Safety Vest', timestamp: '14:21:30', image: '/api/placeholder/120/160' },
    { id: 4, person: 'Person D', violation: 'No Gloves', timestamp: '14:20:12', image: '/api/placeholder/120/160' }
  ];

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-md p-4 mb-6">
          <div className="flex justify-between items-center">
            <h1 className="text-2xl font-bold text-gray-800 flex items-center">
              <AlertTriangle className="mr-2 text-red-500" />
              PPE Safety Monitoring System
            </h1>
            <div className="flex items-center text-gray-600">
              <Clock className="mr-2" size={20} />
              <span className="text-lg font-mono">
                {currentTime.toLocaleTimeString()}
              </span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Live Camera Feed */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow-md p-4">
              <h2 className="text-xl font-semibold mb-4 flex items-center">
                <Camera className="mr-2" />
                Live Camera Feed - Zone A
              </h2>
              <div className="relative">
                <img 
                  src="/api/placeholder/800/450" 
                  alt="Live camera feed" 
                  className="w-full rounded-lg border-2 border-gray-300"
                />
                {/* Detection overlays */}
                <div className="absolute top-4 left-4 bg-red-500 text-white px-2 py-1 rounded text-sm font-semibold">
                  🚫 No Helmet Detected
                </div>
                <div className="absolute top-16 left-4 bg-orange-500 text-white px-2 py-1 rounded text-sm font-semibold">
                  🚫 No Mask Detected
                </div>
                <div className="absolute bottom-4 right-4 bg-black bg-opacity-50 text-white px-3 py-2 rounded">
                  <div className="text-sm">People in Area: {totalPeople}</div>
                  <div className="text-sm">PPE Violations: {totalMissingPPE}</div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Panel */}
          <div className="space-y-6">
            {/* Detection History */}
            <div className="bg-white rounded-lg shadow-md p-4">
              <h3 className="text-lg font-semibold mb-4">Detection History</h3>
              <div className="space-y-3">
                {detectionHistory.map((item, index) => (
                  <div key={index} className="flex justify-between items-center p-3 bg-red-50 rounded-lg border-l-4 border-red-500">
                    <span className="font-medium text-red-800">{item.type}</span>
                    <span className="bg-red-500 text-white px-2 py-1 rounded-full text-sm font-bold">
                      {item.count}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Severity Indicator */}
            <div className="bg-white rounded-lg shadow-md p-4">
              <h3 className="text-lg font-semibold mb-4">Safety Level</h3>
              <div className="text-center">
                <div className={`${severity.color} rounded-lg p-6 mb-4`}>
                  <div className="text-white text-3xl font-bold mb-2">
                    {severity.text}
                  </div>
                  <div className="text-white text-xl">
                    {compliancePercentage}% Compliance
                  </div>
                </div>
                <div className="text-sm text-gray-600">
                  <div>Total People: {totalPeople}</div>
                  <div>Missing PPE: {totalMissingPPE}</div>
                  <div>Compliant: {totalPeople - totalMissingPPE}</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Violation Screenshots Section */}
        <div className="mt-6 bg-white rounded-lg shadow-md p-4">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <User className="mr-2" />
            Recent Violations - Person Screenshots
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {violationScreenshots.map((violation) => (
              <div key={violation.id} className="border rounded-lg p-3 bg-gray-50">
                <img 
                  src={violation.image} 
                  alt={`Violation by ${violation.person}`}
                  className="w-full rounded mb-2 border"
                />
                <div className="text-sm">
                  <div className="font-semibold text-gray-800">{violation.person}</div>
                  <div className="text-red-600 font-medium">{violation.violation}</div>
                  <div className="text-gray-500">{violation.timestamp}</div>
                </div>
                <button className="mt-2 w-full bg-blue-500 text-white py-1 px-2 rounded text-xs hover:bg-blue-600">
                  Send Alert
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Status Bar */}
        <div className="mt-6 bg-white rounded-lg shadow-md p-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-4">
              <div className="flex items-center">
                <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
                <span className="text-sm">System Online</span>
              </div>
              <div className="flex items-center">
                <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
                <span className="text-sm">Camera Connected</span>
              </div>
              <div className="flex items-center">
                <div className="w-3 h-3 bg-yellow-500 rounded-full mr-2"></div>
                <span className="text-sm">{totalMissingPPE} Active Alerts</span>
              </div>
            </div>
            <div className="text-sm text-gray-600">
              Last Updated: {currentTime.toLocaleTimeString()}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PPEAlarmInterface;