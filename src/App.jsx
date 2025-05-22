import React from "react";
import "./App.scss"; // You might want to clear this out or adapt later
import VisualizationCanvas from "./components/TrackVisualizer"; // We'll create this next
import TrackVisualizer from "./components/TrackVisualizer";

const App = () => {
  return (
    <div className="App">
      {/* The titlebar header can remain if you like its styling */}
      <header className="titlebar"></header>
      <div className="main-content-area"> {/* Adjusted class name for clarity */}
        <TrackVisualizer />
      </div>
      {/* Footer removed for now, can be added back if needed */}
    </div>
  );
};

export default App;