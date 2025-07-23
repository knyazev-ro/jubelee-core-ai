#!/bin/sh

set -e

# Запускаем ollama в фоне, чтобы выполнить pull
ollama serve &
sleep 5

# Загружаем модель, если её ещё нет
if ! ollama list | grep -q "gemma3"; then
  echo "Модель не найдена, загружаю..."
  ollama pull gemma3:1b
else
  echo "Модель уже есть"
fi

# Ждём завершения фоновых процессов, затем
# запускаем ollama "по-нормальному" в качестве главного процесса
# Альтернатива: не использовать & вообще
wait
exec ollama serve
