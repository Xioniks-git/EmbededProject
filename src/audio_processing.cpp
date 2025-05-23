#include "audio_processing.h"
#include <math.h>

// Применение окна Ханна
void applyHannWindow(float* buffer, int size) {
    for (int i = 0; i < size; i++) {
        buffer[i] *= 0.5f * (1.0f - cosf(2.0f * PI * i / (size - 1)));
    }
}

// Вычисление FFT (упрощенная версия)
void computeFFT(float* buffer, int size) {
    float real[size];
    float imag[size];
    
    // Копирование входных данных
    for (int i = 0; i < size; i++) {
        real[i] = buffer[i];
        imag[i] = 0;
    }
    
    // Вычисление FFT
    for (int stage = 1; stage <= log2(size); stage++) {
        int m = 1 << stage;
        float wm_real = cosf(2 * PI / m);
        float wm_imag = -sinf(2 * PI / m);
        
        for (int k = 0; k < size; k += m) {
            float w_real = 1;
            float w_imag = 0;
            
            for (int j = 0; j < m/2; j++) {
                float t_real = w_real * real[k + j + m/2] - w_imag * imag[k + j + m/2];
                float t_imag = w_real * imag[k + j + m/2] + w_imag * real[k + j + m/2];
                
                real[k + j + m/2] = real[k + j] - t_real;
                imag[k + j + m/2] = imag[k + j] - t_imag;
                real[k + j] += t_real;
                imag[k + j] += t_imag;
                
                float w_next_real = w_real * wm_real - w_imag * wm_imag;
                float w_next_imag = w_real * wm_imag + w_imag * wm_real;
                w_real = w_next_real;
                w_imag = w_next_imag;
            }
        }
    }
    
    // Вычисление магнитуд
    for (int i = 0; i < size/2; i++) {
        buffer[i] = sqrtf(real[i] * real[i] + imag[i] * imag[i]);
    }
}

// Преобразование частот в мель-шкалу
float hzToMel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

// Преобразование мель-шкалы в частоты
float melToHz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// Вычисление мель-фильтров
void computeMelFilterbank(float* fft_magnitudes, float* mel_energies) {
    float mel_min = hzToMel(MIN_FREQ);
    float mel_max = hzToMel(MAX_FREQ);
    float mel_step = (mel_max - mel_min) / (NUM_MELS + 1);
    
    // Создание точек мель-фильтров
    float mel_points[NUM_MELS + 2];
    for (int i = 0; i < NUM_MELS + 2; i++) {
        mel_points[i] = mel_min + i * mel_step;
    }
    
    // Преобразование точек мель-фильтров в частоты
    float freq_points[NUM_MELS + 2];
    for (int i = 0; i < NUM_MELS + 2; i++) {
        freq_points[i] = melToHz(mel_points[i]);
    }
    
    // Преобразование частот в индексы FFT
    int fft_indices[NUM_MELS + 2];
    for (int i = 0; i < NUM_MELS + 2; i++) {
        fft_indices[i] = roundf(freq_points[i] * FFT_SIZE / SAMPLE_RATE);
    }
    
    // Применение мель-фильтров
    for (int i = 0; i < NUM_MELS; i++) {
        mel_energies[i] = 0;
        for (int j = fft_indices[i]; j < fft_indices[i + 2]; j++) {
            if (j < FFT_SIZE/2) {
                float weight = 1.0f;
                if (j < fft_indices[i + 1]) {
                    weight = (float)(j - fft_indices[i]) / (fft_indices[i + 1] - fft_indices[i]);
                } else {
                    weight = (float)(fft_indices[i + 2] - j) / (fft_indices[i + 2] - fft_indices[i + 1]);
                }
                mel_energies[i] += fft_magnitudes[j] * weight;
            }
        }
    }
}

// Нормализация спектрограммы
void normalizeSpectrogram(float* spectrogram, int size) {
    float max_val = 0;
    for (int i = 0; i < size; i++) {
        if (spectrogram[i] > max_val) {
            max_val = spectrogram[i];
        }
    }
    
    if (max_val > 0) {
        for (int i = 0; i < size; i++) {
            spectrogram[i] /= max_val;
        }
    }
}

// Основная функция преобразования аудио в мель-спектрограмму
void audioToMelSpectrogram(float* audio, float* spectrogram) {
    float fft_buffer[FFT_SIZE];
    float mel_energies[NUM_MELS];
    
    for (int frame = 0; frame < NUM_FRAMES; frame++) {
        // Копирование и применение окна
        for (int i = 0; i < FFT_SIZE; i++) {
            int idx = frame * HOP_LENGTH + i;
            fft_buffer[i] = (idx < BUFFER_SIZE) ? audio[idx] : 0;
        }
        applyHannWindow(fft_buffer, FFT_SIZE);
        
        // Вычисление FFT
        computeFFT(fft_buffer, FFT_SIZE);
        
        // Применение мель-фильтров
        computeMelFilterbank(fft_buffer, mel_energies);
        
        // Копирование результатов в спектрограмму
        for (int mel = 0; mel < NUM_MELS; mel++) {
            spectrogram[mel * NUM_FRAMES + frame] = mel_energies[mel];
        }
    }
    
    // Нормализация всей спектрограммы
    normalizeSpectrogram(spectrogram, NUM_MELS * NUM_FRAMES);
} 