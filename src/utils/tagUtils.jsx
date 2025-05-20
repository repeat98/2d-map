// src/components/tagUtils.js

const saturation = 100; // Saturation percentage (0-100)
const luminance = 95; // Luminance percentage (0-100)
const textSaturation = 40; // Text saturation percentage
const lightTextLuminance = 20; // Text luminance for dark backgrounds
const darkTextLuminance = 50; 

// Function to calculate luminance from HSL values
function calculateLuminance(hue, sat, lum) {
  const a = (sat / 100) * (lum / 100);
  const l = lum / 100;
  return 0.2126 * a + 0.7152 * l + 0.0722 * l;
}

// Function to generate HSL color string
function getHSLColor(hue, sat, lum) {
  return `hsl(${hue}, ${sat}%, ${lum}%)`;
}

// Function to generate a deterministic hue from a string
function stringToHue(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  return Math.abs(hash % 360);
}

// Function to get colors for tags
export function getTagColors(tag) {
  const hue = stringToHue(tag);
  const backgroundColor = getHSLColor(hue, saturation, luminance);

  // Calculate luminance of background color
  const bgLuminance = calculateLuminance(hue, saturation, luminance);

  // Choose text color based on background luminance
  const textLuminance =
    bgLuminance > 0.5 ? darkTextLuminance : lightTextLuminance;
  const textColor = getHSLColor(hue, textSaturation, textLuminance);

  return { backgroundColor, textColor };
}