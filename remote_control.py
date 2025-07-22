import serial
import argparse
from loguru import logger


class RemoteControl():
    """Класс для управления ИК-портом."""
    
    tx_port: str = '/dev/ttyS5'
    rx_port: str = '/dev/ttyS4'
    def __init__(self):
        self.tv_code: str = 'a1 f1 01 01 a2'  # Пример кода для отправки
        self.audio_code: str = 'a1 f1 01 02 a2'  # Пример кода для отправки аудио

    
    def command_router(self, cmd: str | None = None) -> str:
        """Маршрутизатор команд для отправки ИК-команд."""
        match cmd:
            case 'tv':
                output = self._send_ir(self.tv_code)
            case 'audio':
                output = self._send_ir(self.audio_code)
            case None:
                output = self._send_ir(self.tv_code)
                output = self._send_ir(self.audio_code)
            case _:
                output = f"❌ Неизвестная команда: {cmd}"
        
        return output


    def _send_ir(self, hex_code_string: str) -> str:
        """Отправляет ИК-команду, представленную в виде hex-строки."""
        try:
            # Преобразуем строку hex в байты
            # Например, 'a1 f1 01 01 a2' -> b'\xa1\xf1\x01\x01\xa2'
            command_bytes = bytes.fromhex(hex_code_string)

            with serial.Serial(self.tx_port, 9600, timeout=1) as ser:
                print(f"⚠️ Отправка команды: {hex_code_string}")
                ser.write(command_bytes)
                output = "✅ Команда отправлена."

        except serial.SerialException as e:
            output = "❌ Ошибка: Не удалось открыть порт"
            logger.error(output, exc_info=e)
        except ValueError:
            output = "❌ Ошибка: Неверный формат hex-строки. Используйте пробелы в качестве разделителя."
            logger.error(output)
        
        return output
    

    @classmethod
    def _read_ir(cls) -> None:
        """Читает данные из ИК-порта."""
        try:
            with serial.Serial(cls.rx_port, 9600, timeout=1) as ser:
                print("⚠️ Направьте пульт на ИК-приемник и нажмите кнопку...")
                while True:
                    ir_code = ser.read(5)
                    if ir_code:
                        print(ir_code.hex(' '))
        except serial.SerialException as e:
            print("❌ Ошибка: Не удалось открыть порт")
            logger.error(e)
        except KeyboardInterrupt:
            print("\n⚠️ Программа завершена.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IR Remote Control')
    parser.add_argument('--read', action='store_true', help='Read IR codes')
    args = parser.parse_args()
    if args.read:
        RemoteControl._read_ir()
        # python remote_control.py --read