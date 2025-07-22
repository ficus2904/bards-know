# /// script
# dependencies = [
#   "pyserial",
#   "loguru",
# ]
# ///

import serial
import argparse
from loguru import logger


class RemoteControl():
    """Класс для управления ИК-портом."""
    
    port: str = '/dev/ttyS5'
    def __init__(self):
        self.tv_switch: str =    'A1 F1 04 fb 08'
        self.audio_switch: str = 'A1 F1 02 fd 48'

    
    def command_router(self, cmd: str | None = None) -> str:
        """Маршрутизатор команд для отправки ИК-команд."""
        match cmd:
            case 'tv':
                output = self._send_ir(self.tv_switch)
            case 'audio':
                output = self._send_ir(self.audio_switch)
            case None:
                output = self._send_ir(self.tv_switch)
                output = self._send_ir(self.audio_switch)
            case _:
                output = f"❌ Неизвестная команда: {cmd}"
        
        return output


    def _send_ir(self, hex_code_string: str) -> str:
        """Отправляет ИК-команду, представленную в виде hex-строки."""
        try:
            command_bytes = bytes.fromhex(hex_code_string)

            with serial.Serial(self.port, 9600, timeout=1) as ser:
                logger.info(f"⚠️ Отправка команды: {hex_code_string}")
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
            with serial.Serial(cls.port, 9600, timeout=1) as ser:
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


    @classmethod
    def _test(cls) -> bool:
        # Собираем полную команду
        full_command = 'A1 F1 04 fb 08'
        print(f"⚠️ Send: '{full_command}'...")
        try:
            command_bytes = bytes.fromhex(full_command.replace(' ', ''))
            
            if len(command_bytes) != 5:
                print(f"❌ Ошибка сборки, итоговая команда не 5 байт.")
                return False

            with serial.Serial(cls.port) as ser:
                ser.write(command_bytes)

            print("✅ Success!")

        except Exception as e:
            print(f"❌Произошла ошибка при отправке: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IR Remote Control')
    parser.add_argument('--read', action='store_true', help='Read IR codes')
    parser.add_argument('--send', action='store_true', help='Send IR codes')
    args = parser.parse_args()
    if args.read:
        RemoteControl._read_ir()
    elif args.send:
        RemoteControl._test()
        # uv run remote_control.py --read
    ## add to docker-compose.yml
    # devices:
    #   - "/dev/ttyS5:/dev/ttyS5"
    # group_add:
    #   - "20" 
    ## and pyserial to requirements.txt