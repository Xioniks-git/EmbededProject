#include <Arduino.h>
#include "driver/i2s.h"
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"  // Будет создан автоматически из .tflite файла
#include "audio_processing.h"

// Дополнительные константы для аудио
const int SAMPLE_BITS = 16;
const int CHANNELS = 1;
const int SPECTROGRAM_SIZE = 1960;  // 40 * 49 * 1 (обновлено под новую модель)

// Буферы для аудио
int16_t sampleBuffer[BUFFER_SIZE];
float audioBuffer[BUFFER_SIZE];
float spectrogram[SPECTROGRAM_SIZE];
// int8_t quantized_spectrogram[SPECTROGRAM_SIZE];  // Убрано - не нужно для float32

// Глобальные переменные для TensorFlow Lite
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Буфер для TensorFlow Lite
constexpr int kTensorArenaSize = 200 * 1024;  // Увеличиваем для float32 модели
uint8_t* tensor_arena = nullptr;  // Будет выделен в PSRAM

// Имена классов
const char* class_names[] = {"Разбитие стекла", "Открытие двери", "Скрип пола"};

// Конфигурация I2S для PDM микрофона (обновленная)
const i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = (i2s_bits_per_sample_t)SAMPLE_BITS,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,  // Уменьшаем количество буферов
    .dma_buf_len = 256,  // Увеличиваем размер буфера
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
};

// Конфигурация пинов I2S для XIAO ESP32S3
const i2s_pin_config_t pin_config = {
    .mck_io_num = I2S_PIN_NO_CHANGE,
    .bck_io_num = I2S_PIN_NO_CHANGE,  // PDM Clock - встроенный
    .ws_io_num = I2S_PIN_NO_CHANGE,   // PDM Data - встроенный
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_PIN_NO_CHANGE  // Используем встроенный PDM микрофон
};

void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);
    
    Serial.println("Инициализация...");
    
    // Проверка наличия PSRAM
    if (!psramFound()) {
        Serial.println("Ошибка: PSRAM не найден!");
        return;
    }
    
    // Выделение памяти для TensorFlow в PSRAM
    tensor_arena = (uint8_t*)ps_malloc(kTensorArenaSize);
    if (tensor_arena == nullptr) {
        Serial.println("Ошибка выделения памяти для TensorFlow!");
        return;
    }
    
    // Инициализация I2S для PDM микрофона
    esp_err_t err = i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        Serial.println("Ошибка инициализации I2S!");
        return;
    }
    
    err = i2s_set_pin(I2S_NUM_0, &pin_config);
    if (err != ESP_OK) {
        Serial.println("Ошибка настройки пинов I2S!");
        return;
    }
    
    // Загрузка модели
    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Несовместимая версия схемы модели!");
        return;
    }
    
    // Создание интерпретатора со всеми операциями
    static tflite::AllOpsResolver resolver;
    
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;
    
    // Выделение тензоров
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("Ошибка выделения тензоров!");
        return;
    }
    
    // Получение указателей на входной и выходной тензоры
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    // Проверка входного тензора
    if (input == nullptr) {
        Serial.println("Ошибка: входной тензор не найден!");
        return;
    }
    
    // Вывод подробной информации о модели и тензорах
    Serial.println("\nИнформация о модели:");
    Serial.print("Количество операций: ");
    Serial.println(model->subgraphs()->Get(0)->operators()->size());
    
    Serial.println("\nИнформация о входном тензоре:");
    Serial.print("Тип данных: ");
    switch (input->type) {
        case kTfLiteFloat32:
            Serial.println("kTfLiteFloat32");
            break;
        case kTfLiteInt8:
            Serial.println("kTfLiteInt8");
            break;
        case kTfLiteUInt8:
            Serial.println("kTfLiteUInt8");
            break;
        default:
            Serial.println("Другой");
    }
    
    Serial.print("Размеры: [");
    for (int i = 0; i < input->dims->size; i++) {
        Serial.print(input->dims->data[i]);
        if (i < input->dims->size - 1) Serial.print(", ");
    }
    Serial.println("]");
    
    // Получение параметров квантования
    Serial.println("\nПараметры входного тензора:");
    if (input->quantization.type == kTfLiteAffineQuantization) {
        Serial.println("Квантование обнаружено (но не используется для float32)");
    } else {
        Serial.println("Квантование НЕ используется - входные данные float32");
    }
    
    // Информация о выходном тензоре
    Serial.println("\nИнформация о выходном тензоре:");
    Serial.print("Тип: ");
    switch (output->type) {
        case kTfLiteFloat32:
            Serial.println("kTfLiteFloat32");
            break;
        case kTfLiteInt8:
            Serial.println("kTfLiteInt8");
            break;
        default:
            Serial.println("Другой");
    }
    Serial.print("Размеры: [");
    for (int i = 0; i < output->dims->size; i++) {
        Serial.print(output->dims->data[i]);
        if (i < output->dims->size - 1) Serial.print(", ");
    }
    Serial.println("]");
    
    Serial.println("\nКлассы для распознавания:");
    for (int i = 0; i < 3; i++) {
        Serial.print(i); Serial.print(": "); Serial.println(class_names[i]);
    }
    
    // Тестирование микрофона
    Serial.println("\n=== ТЕСТИРОВАНИЕ МИКРОФОНА ===");
    Serial.println("Тестируем I2S и PDM микрофон...");
    
    size_t test_bytes_read = 0;
    int16_t test_buffer[256];
    esp_err_t test_err = i2s_read(I2S_NUM_0, test_buffer, sizeof(test_buffer), &test_bytes_read, 1000);
    
    if (test_err == ESP_OK && test_bytes_read > 0) {
        int16_t test_max = 0, test_min = 0;
        int test_non_zero = 0;
        
        for (int i = 0; i < 256; i++) {
            if (test_buffer[i] > test_max) test_max = test_buffer[i];
            if (test_buffer[i] < test_min) test_min = test_buffer[i];
            if (test_buffer[i] != 0) test_non_zero++;
        }
        
        Serial.print("Тест успешен! Прочитано: "); Serial.print(test_bytes_read); Serial.println(" байт");
        Serial.print("Диапазон значений: "); Serial.print(test_min); Serial.print(" до "); Serial.println(test_max);
        Serial.print("Ненулевых значений: "); Serial.print(test_non_zero); Serial.println("/256");
        
        if (test_non_zero > 10 && (test_max != test_min)) {
            Serial.println("✅ Микрофон работает корректно!");
        } else {
            Serial.println("⚠️  Микрофон может работать некорректно - данные статичны");
        }
    } else {
        Serial.print("❌ Ошибка тестирования микрофона: ");
        Serial.println(esp_err_to_name(test_err));
    }
    
    Serial.println("\nИнициализация завершена!");
    Serial.println("Начинаю прослушивание звуков...");
    Serial.println("Попробуйте издать один из обученных звуков:");
    Serial.println("- Разбить стекло (или постучать по стеклу)");
    Serial.println("- Открыть/закрыть дверь");
    Serial.println("- Скрипнуть половицей или мебелью");
    Serial.println("=====================================\n");
}

void loop() {
    size_t bytes_read = 0;
    esp_err_t err = i2s_read(I2S_NUM_0, sampleBuffer, BUFFER_SIZE * sizeof(int16_t), &bytes_read, portMAX_DELAY);
    
    if (err == ESP_OK && bytes_read > 0) {
        // Детальная диагностика аудио потока
        int16_t max_sample = 0;
        int16_t min_sample = 0;
        int32_t sum = 0;
        int non_zero_count = 0;
        
        for (int i = 0; i < BUFFER_SIZE; i++) {
            if (sampleBuffer[i] > max_sample) max_sample = sampleBuffer[i];
            if (sampleBuffer[i] < min_sample) min_sample = sampleBuffer[i];
            sum += sampleBuffer[i];
            if (sampleBuffer[i] != 0) non_zero_count++;
        }
        
        float average = (float)sum / BUFFER_SIZE;
        
        Serial.print("\n=== ДИАГНОСТИКА АУДИО ===");
        Serial.print("\nПрочитано байт: "); Serial.println(bytes_read);
        Serial.print("Размер буфера: "); Serial.println(BUFFER_SIZE);
        Serial.print("Max sample: "); Serial.println(max_sample);
        Serial.print("Min sample: "); Serial.println(min_sample);
        Serial.print("Среднее: "); Serial.println(average, 2);
        Serial.print("Ненулевых сэмплов: "); Serial.print(non_zero_count);
        Serial.print(" из "); Serial.println(BUFFER_SIZE);
        
        // Проверка вариативности данных
        bool data_varies = (max_sample != min_sample) && (non_zero_count > BUFFER_SIZE / 10);
        Serial.print("Данные изменяются: "); Serial.println(data_varies ? "ДА" : "НЕТ");
        
        if (!data_varies) {
            Serial.println("⚠️  ПРОБЛЕМА: Аудио данные статичны или отсутствуют!");
            Serial.println("Попробуйте:");
            Serial.println("1. Издать громкий звук рядом с микрофоном");
            Serial.println("2. Проверить подключение микрофона");
            delay(1000);
            return;
        }
        
        // Преобразование аудио в мель-спектрограмму
        for (int i = 0; i < BUFFER_SIZE; i++) {
            audioBuffer[i] = sampleBuffer[i] / 32768.0f;
        }
        
        Serial.println("\nВычисляем спектрограмму...");
        audioToMelSpectrogram(audioBuffer, spectrogram);
        
        // Анализ спектрограммы
        float min_spec = 1000.0f, max_spec = -1000.0f;
        float spec_sum = 0;
        int non_zero_spec = 0;
        
        for (int i = 0; i < SPECTROGRAM_SIZE; i++) {
            if (spectrogram[i] < min_spec) min_spec = spectrogram[i];
            if (spectrogram[i] > max_spec) max_spec = spectrogram[i];
            spec_sum += spectrogram[i];
            if (spectrogram[i] > 0.001f) non_zero_spec++;
        }
        
        float spec_avg = spec_sum / SPECTROGRAM_SIZE;
        
        Serial.println("=== АНАЛИЗ СПЕКТРОГРАММЫ ===");
        Serial.print("Min: "); Serial.println(min_spec, 4);
        Serial.print("Max: "); Serial.println(max_spec, 4);
        Serial.print("Среднее: "); Serial.println(spec_avg, 4);
        Serial.print("Значимых значений: "); Serial.print(non_zero_spec);
        Serial.print(" из "); Serial.println(SPECTROGRAM_SIZE);
        
        // Проверяем тип входного тензора
        if (input->type == kTfLiteFloat32) {
            Serial.println("\nКопируем float32 данные...");
            memcpy(input->data.f, spectrogram, SPECTROGRAM_SIZE * sizeof(float));
        } else {
            Serial.print("Неожиданный тип входного тензора: ");
            Serial.println(input->type);
            return;
        }

        // Запуск инференса
        Serial.println("Запуск инференса...");
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
            Serial.println("Ошибка инференса!");
            return;
        }

        // Получение результатов
        float scores[3] = {0, 0, 0};
        float max_score = -1000.0f;
        int max_index = 0;
        
        for (int i = 0; i < 3; i++) {
            scores[i] = output->data.f[i];
            if (scores[i] > max_score) {
                max_score = scores[i];
                max_index = i;
            }
        }

        // Вывод результатов
        Serial.println("\n=== РЕЗУЛЬТАТЫ РАСПОЗНАВАНИЯ ===");
        for (int i = 0; i < 3; i++) {
            Serial.print("  "); Serial.print(class_names[i]); 
            Serial.print(": "); Serial.println(scores[i], 4);
        }
        
        Serial.print("\n🎯 РАСПОЗНАННЫЙ ЗВУК: ");
        Serial.print(class_names[max_index]);
        Serial.print(" (уверенность: ");
        Serial.print(max_score, 4);
        Serial.println(")");
        
        // Анализ уверенности
        if (max_score < 0.3f) {
            Serial.println("❓ Очень низкая уверенность - возможно, неизвестный звук");
        } else if (max_score < 0.6f) {
            Serial.println("⚠️  Низкая уверенность - нужен более четкий звук");
        } else {
            Serial.println("✅ Высокая уверенность в распознавании!");
        }
        
        Serial.println("==============================");
        delay(2000); // Увеличиваем паузу для лучшей читаемости
    } else {
        Serial.print("Ошибка чтения I2S: ");
        Serial.println(esp_err_to_name(err));
        delay(1000);
    }
} 