import asyncio
from pathlib import Path

import telegram as tg

TELEGRAM_BOT_TOKEN = "7497996082:AAEGQin96OvPZ1afUD2X_iz--WYYiaW2JEs"
TELEGRAM_USER_ID = 284330085
bot = tg.Bot(token=TELEGRAM_BOT_TOKEN)


with Path("/home/akarnachev/src/shaderbox/new_low.webm").open("rb") as f:
    data = f.read()

stickers = [
    tg.InputSticker(
        sticker=data,
        emoji_list=["😺"],
        format=tg.constants.StickerFormat.VIDEO,
    )
]

loop = asyncio.new_event_loop()

loop.run_until_complete(bot.initialize())

loop.run_until_complete(
    bot.create_new_sticker_set(
        user_id=TELEGRAM_USER_ID,
        name=f"test_by_{bot.username}",
        title="ShaderBox",
        stickers=stickers,
        sticker_type=tg.Sticker.REGULAR,
        needs_repainting=False,
    )
)
