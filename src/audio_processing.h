#ifndef AUDIO_PROCESSING_H
#define AUDIO_PROCESSING_H

#include <Arduino.h>

// Константы для обработки аудио
const int SAMPLE_RATE = 16000;
const int FFT_SIZE = 512;
const int NUM_MELS = 40;
const int NUM_FRAMES = 49;
const int HOP_LENGTH = 160;
const int BUFFER_SIZE = NUM_FRAMES * HOP_LENGTH + FFT_SIZE;
const int MIN_FREQ = 20;
const int MAX_FREQ = 8000;

// Функции обработки аудио
void applyHannWindow(float* buffer, int size);
void computeFFT(float* buffer, int size);
float hzToMel(float hz);
float melToHz(float mel);
void computeMelFilterbank(float* fft_magnitudes, float* mel_energies);
void normalizeSpectrogram(float* spectrogram, int size);
void audioToMelSpectrogram(float* audio, float* spectrogram);

#endif // AUDIO_PROCESSING_H 