"""Messaging bridge CLI — start a bridge to connect a chat platform to a running agent.

Usage:
    python -m messaging.cli --platform telegram --token BOT_TOKEN
    python -m messaging.cli --platform discord --token BOT_TOKEN
    python -m messaging.cli --platform telegram --token BOT_TOKEN --host 127.0.0.1 --port 8080
"""

import argparse
import asyncio
import logging
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Motherlabs messaging bridge — connect chat platforms to agent runtime",
    )
    parser.add_argument(
        "--platform", "-p",
        choices=["telegram", "discord"],
        required=True,
        help="Chat platform to bridge",
    )
    parser.add_argument(
        "--token", "-t",
        default=os.environ.get("BOT_TOKEN", ""),
        help="Bot token (or set BOT_TOKEN env var)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("RUNTIME_HOST", "127.0.0.1"),
        help="Runtime TCP host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("RUNTIME_PORT", "8080")),
        help="Runtime TCP port (default: 8080)",
    )
    parser.add_argument(
        "--target",
        default="Chat Agent",
        help="Default component target for messages (default: 'Chat Agent')",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )

    args = parser.parse_args()

    if not args.token:
        print("Error: Bot token required. Use --token or set BOT_TOKEN env var.", file=sys.stderr)
        sys.exit(1)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if args.platform == "telegram":
        from messaging.telegram_bridge import TelegramBridge
        bridge = TelegramBridge(
            token=args.token,
            tcp_host=args.host,
            tcp_port=args.port,
            default_target=args.target,
        )
    elif args.platform == "discord":
        from messaging.discord_bridge import DiscordBridge
        bridge = DiscordBridge(
            token=args.token,
            tcp_host=args.host,
            tcp_port=args.port,
            default_target=args.target,
        )
    else:
        print(f"Unknown platform: {args.platform}", file=sys.stderr)
        sys.exit(1)

    print(f"Starting {args.platform} bridge → {args.host}:{args.port}")

    try:
        asyncio.run(bridge.start())
    except KeyboardInterrupt:
        print("\nBridge stopped.")
    except Exception as e:
        print(f"Bridge error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
