// Tag.jsx
import React from 'react';
import './Tag.scss';
import PropTypes from 'prop-types';

const tagColorMap = new Map();
const saturation = 50;
const luminance = 50;
const textSaturation = 100;
const darkTextLuminance = 20;
const lightTextLuminance = 80;

export function getTagColors(tag) {
  if (!tagColorMap.has(tag)) {
    const index = tagColorMap.size;
    const hue = ((index * 360) / 12) % 360; // Adjust cycle range as needed
    const backgroundColor = getHSLColor(hue, saturation, luminance);
    const bgLuminance = calculateLuminance(hue, saturation, luminance);
    const textLuminance = bgLuminance > 0.5 ? darkTextLuminance : lightTextLuminance;
    const textColor = getHSLColor(hue, textSaturation, textLuminance);

    tagColorMap.set(tag, { backgroundColor, textColor });
  }
  return tagColorMap.get(tag);
}

function calculateLuminance(hue, sat, lum) {
  const a = (sat / 100) * (lum / 100);
  const l = lum / 100;
  return 0.2126 * a + 0.7152 * l + 0.0722 * l;
}

function getHSLColor(hue, sat, lum) {
  return `hsl(${hue}, ${sat}%, ${lum}%)`;
}

const Tag = ({ tag }) => {
  const { backgroundColor, textColor } = getTagColors(tag);

  return (
    <div className="tag" style={{ backgroundColor, color: textColor }}>
      {tag}
    </div>
  );
};

Tag.propTypes = {
  tag: PropTypes.string.isRequired,
};

export default Tag;