// src/components/Activity.jsx
import React from 'react';
import './Activity.scss';

const Activity = () => {
  return (
    <div className="activity">
      <progress id="progressBar" value="0" max="100"></progress>
      <p id="progressText">0%</p>
    </div>
  );
};

export default Activity;