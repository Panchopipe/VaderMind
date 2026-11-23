"""Pair VaderMind with a Samsung Tizen TV and keep the token outside Git.

The TV shows an "allow this device" prompt the first time; once someone accepts
it with the remote, the TV hands back a token that later connections reuse
without asking again. This script only opens the remote-control channel and
waits for that token: it sends no keys and changes nothing else on the TV.

Usage (laptop or Pi, from the VaderMind folder):
    python scripts/samsung_tv_pair.py 192.0.2.23 profiles/pablo/samsung_tv_token

Running it again with a saved token only checks that the token still works.
The token is never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import ssl
import sys
from pathlib import Path

CLIENT_NAME = "VaderMind"
WAIT_SECONDS = 60


def channel_url(host: str, token: str | None) -> str:
    name = base64.b64encode(CLIENT_NAME.encode("utf-8")).decode("ascii")
    url = f"wss://{host}:8002/api/v2/channels/samsung.remote.control?name={name}"
    if token:
        url += f"&token={token}"
    return url


async def pair(host: str, token_path: Path) -> int:
    import aiohttp

    saved = token_path.read_text(encoding="utf-8").strip() if token_path.exists() else None
    # The TV uses a self-signed certificate; this connection only goes to the
    # TV address given on the command line, inside the home network.
    insecure = ssl.create_default_context()
    insecure.check_hostname = False
    insecure.verify_mode = ssl.CERT_NONE

    if saved:
        print("Ya hay una clave guardada: compruebo que siga sirviendo.")
    else:
        print(f"Mira el TV y acepta 'VaderMind' con el control (tienes {WAIT_SECONDS} s).")

    timeout = aiohttp.ClientTimeout(total=WAIT_SECONDS + 10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.ws_connect(channel_url(host, saved), ssl=insecure) as ws:
                while True:
                    message = await ws.receive(timeout=WAIT_SECONDS)
                    if message.type != aiohttp.WSMsgType.TEXT:
                        print(f"El TV cerro la conexion ({message.type.name}): no se acepto el permiso.")
                        return 1
                    data = json.loads(message.data)
                    event = data.get("event", "")
                    if event == "ms.channel.connect":
                        token = (data.get("data") or {}).get("token")
                        if token and token != saved:
                            token_path.parent.mkdir(parents=True, exist_ok=True)
                            token_path.write_text(str(token), encoding="utf-8")
                            print(f"Emparejado. Clave guardada en {token_path} ({len(str(token))} caracteres).")
                        elif saved:
                            print("La clave guardada sigue funcionando.")
                        else:
                            print("Conectado, pero el TV no entrego clave (puede no pedir permiso).")
                        return 0
                    if event == "ms.channel.unauthorized":
                        print("El TV rechazo el permiso (no aceptado o denegado).")
                        return 1
                    if event == "ms.channel.timeOut":
                        print("Se acabo el tiempo del aviso en el TV. Vuelve a intentarlo.")
                        return 1
        except (asyncio.TimeoutError, TimeoutError):
            print("Nadie acepto el aviso a tiempo. Vuelve a intentarlo.")
            return 1
        except aiohttp.ClientError as error:
            print(f"No pude conectar con el TV en {host}:8002: {error}")
            return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("host", help="IP del TV Samsung")
    parser.add_argument("token_path", type=Path, help="archivo donde guardar la clave (fuera de Git)")
    args = parser.parse_args()
    return asyncio.run(pair(args.host, args.token_path))


if __name__ == "__main__":
    sys.exit(main())
