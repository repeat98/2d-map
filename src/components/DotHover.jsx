import React, { useEffect, useRef } from 'react';
import * as PIXI from 'pixi.js';
import WaveSurfer from 'wavesurfer.js';
import defaultArtwork from "../../assets/default-artwork.png";
import './DotHover.scss';

const PLAY_BUTTON_COLOR = 0x6A82FB;
const PLAY_BUTTON_HOVER_COLOR = 0x8BA3FF;
const PLAY_BUTTON_SIZE = 24;
const TOOLTIP_WIDTH = 300;
const TOOLTIP_HEIGHT = 200;
const TOOLTIP_PADDING = 10;
const COVER_ART_SIZE = 80;

const DotHover = ({ 
  track, 
  xFeature, 
  yFeature, 
  xFeatureLabel, 
  yFeatureLabel, 
  xValue, 
  yValue, 
  formatTickValue,
  wavesurfer,
  onPlayPause,
  isPlaying,
  position,
  container
}) => {
  const containerRef = useRef(null);
  const playButtonRef = useRef(null);
  const playIconRef = useRef(null);
  const waveformContainerRef = useRef(null);
  const coverArtSpriteRef = useRef(null);
  const titleTextRef = useRef(null);
  const featuresTextRef = useRef(null);

  useEffect(() => {
    if (!container) return;

    // Create container
    const tooltipContainer = new PIXI.Container();
    containerRef.current = tooltipContainer;
    container.addChild(tooltipContainer);

    // Set position
    tooltipContainer.position.set(position.x, position.y);

    // Create background
    const bg = new PIXI.Graphics()
      .roundRect(0, 0, TOOLTIP_WIDTH, TOOLTIP_HEIGHT, 8)
      .fill({ color: 0x333333 });
    tooltipContainer.addChild(bg);

    // Create cover art sprite
    const coverArt = new PIXI.Sprite(PIXI.Texture.EMPTY);
    coverArt.position.set(TOOLTIP_PADDING, TOOLTIP_PADDING);
    coverArt.width = COVER_ART_SIZE;
    coverArt.height = COVER_ART_SIZE;
    tooltipContainer.addChild(coverArt);
    coverArtSpriteRef.current = coverArt;

    // Create title text
    const titleText = new PIXI.Text({
      text: track.title || 'Unknown Title',
      style: {
        fontFamily: 'Arial',
        fontSize: 16,
        fontWeight: 'bold',
        fill: 0xFFFFFF,
        wordWrap: true,
        wordWrapWidth: TOOLTIP_WIDTH - (TOOLTIP_PADDING * 2 + COVER_ART_SIZE + TOOLTIP_PADDING)
      }
    });
    titleText.position.set(TOOLTIP_PADDING + COVER_ART_SIZE + TOOLTIP_PADDING, TOOLTIP_PADDING);
    tooltipContainer.addChild(titleText);
    titleTextRef.current = titleText;

    // Create features text
    const featuresText = new PIXI.Text({
      text: `${xFeatureLabel}: ${formatTickValue(xValue)}\n${yFeatureLabel}: ${formatTickValue(yValue)}`,
      style: {
        fontFamily: 'Arial',
        fontSize: 14,
        fill: 0xAAAAAA,
        wordWrap: true,
        wordWrapWidth: TOOLTIP_WIDTH - (TOOLTIP_PADDING * 2 + COVER_ART_SIZE + TOOLTIP_PADDING)
      }
    });
    featuresText.position.set(TOOLTIP_PADDING + COVER_ART_SIZE + TOOLTIP_PADDING, TOOLTIP_PADDING + 30);
    tooltipContainer.addChild(featuresText);
    featuresTextRef.current = featuresText;

    // Create play button
    const playButton = new PIXI.Graphics()
      .circle(0, 0, PLAY_BUTTON_SIZE / 2)
      .fill({ color: PLAY_BUTTON_COLOR });
    playButton.position.set(TOOLTIP_WIDTH - PLAY_BUTTON_SIZE - 10, PLAY_BUTTON_SIZE + 10);
    playButton.eventMode = 'static';
    playButton.cursor = 'pointer';
    tooltipContainer.addChild(playButton);
    playButtonRef.current = playButton;

    // Create play icon
    const playIcon = new PIXI.Graphics();
    playIcon.beginFill(0xFFFFFF);
    playIcon.moveTo(-4, -6);
    playIcon.lineTo(-4, 6);
    playIcon.lineTo(6, 0);
    playIcon.endFill();
    playButton.addChild(playIcon);
    playIconRef.current = playIcon;

    // Add hover effect for play button
    playButton.on('pointerover', () => {
      playButton.clear()
        .circle(0, 0, PLAY_BUTTON_SIZE / 2)
        .fill({ color: PLAY_BUTTON_HOVER_COLOR });
      playButton.addChild(playIcon);
    });

    playButton.on('pointerout', () => {
      playButton.clear()
        .circle(0, 0, PLAY_BUTTON_SIZE / 2)
        .fill({ color: PLAY_BUTTON_COLOR });
      playButton.addChild(playIcon);
    });

    // Add click handler for play button
    playButton.on('pointerdown', () => {
      onPlayPause();
    });

    // Create waveform container
    const waveformContainer = new PIXI.Container();
    waveformContainer.position.set(TOOLTIP_PADDING, TOOLTIP_PADDING + COVER_ART_SIZE + TOOLTIP_PADDING);
    tooltipContainer.addChild(waveformContainer);
    waveformContainerRef.current = waveformContainer;

    // Load cover art
    const loadCoverArt = async () => {
      try {
        const artworkPath = track.artwork_thumbnail_path || track.coverArtUrl || defaultArtwork;
        const texture = await PIXI.Texture.from(artworkPath);
        coverArt.texture = texture;
      } catch (error) {
        console.error("💥 Error loading cover art:", error);
        coverArt.texture = PIXI.Texture.from(defaultArtwork);
      }
    };
    loadCoverArt();

    // Load waveform
    const loadWaveform = async () => {
      try {
        // Clear previous waveform
        waveformContainer.removeChildren();
        
        // Add background
        const bg = new PIXI.Graphics()
          .rect(0, 0, 150, 40)
          .fill({ color: 0x1A1A1A });
        waveformContainer.addChild(bg);

        if (track.audioUrl) {
          const waveformResponse = await fetch(`http://localhost:3000/tracks/waveform/${track.id}`);
          if (waveformResponse.ok) {
            const waveformData = await waveformResponse.json();
            
            // Draw waveform
            const waveformGraphics = new PIXI.Graphics();
            waveformGraphics.setStrokeStyle({ width: 1, color: 0x6A82FB });
            
            const peaks = waveformData.waveform;
            const width = 150;
            const height = 40;
            const centerY = height / 2;
            const step = width / peaks.length;
            
            waveformGraphics.moveTo(0, centerY);
            
            for (let i = 0; i < peaks.length; i++) {
              const x = i * step;
              const y = centerY + (peaks[i] * centerY);
              waveformGraphics.lineTo(x, y);
            }
            
            waveformContainer.addChild(waveformGraphics);
          }
        }
      } catch (error) {
        console.error("💥 Error loading waveform:", error);
        
        const errorText = new PIXI.Text({
          text: 'Error loading waveform',
          style: {
            fontFamily: 'Arial',
            fontSize: 12,
            fill: 0xFF0000,
            align: 'center'
          }
        });
        errorText.anchor.set(0.5);
        errorText.position.set(75, 20);
        waveformContainer.addChild(errorText);
      }
    };
    loadWaveform();

    return () => {
      if (tooltipContainer.parent) {
        tooltipContainer.parent.removeChild(tooltipContainer);
      }
      tooltipContainer.destroy({ children: true });
    };
  }, [track, xFeature, yFeature, xFeatureLabel, yFeatureLabel, xValue, yValue, formatTickValue, position, container]);

  useEffect(() => {
    if (playIconRef.current) {
      playIconRef.current.clear();
      playIconRef.current.beginFill(0xFFFFFF);
      if (isPlaying) {
        // Draw pause icon
        playIconRef.current.drawRect(-4, -6, 8, 12);
      } else {
        // Draw play icon
        playIconRef.current.moveTo(-4, -6);
        playIconRef.current.lineTo(-4, 6);
        playIconRef.current.lineTo(6, 0);
      }
      playIconRef.current.endFill();
    }
  }, [isPlaying]);

  return null; // This component doesn't render anything directly
};

export default DotHover; 