import os


def ifelse(condition, a, b):
    if condition:
        return a
    else:
        return b


class PrintDecorations:
    RESET = "\u001b[0m"

    # decorations
    BOLD = "\u001b[1m"
    UNDERLINE = "\u001b[4m"
    REVERSED = "\u001b[7m"

    # coloring
    BLACK = "\u001b[30m"
    RED = "\u001b[31m"
    GREEN = "\u001b[32m"
    YELLOW = "\u001b[33m"
    BLUE = "\u001b[34m"
    MAGENTA = "\u001b[35m"
    CYAN = "\u001b[36m"
    WHITE = "\u001b[37m"

    # coloring, bright version
    BLACK_BR = "\u001b[30;1m"
    RED_BR = "\u001b[31;1m"
    GREEN_BR = "\u001b[32;1m"
    YELLOW_BR = "\u001b[33;1m"
    BLUE_BR = "\u001b[34;1m"
    MAGENTA_BR = "\u001b[35;1m"
    CYAN_BR = "\u001b[36;1m"
    WHITE_BR = "\u001b[37;1m"

    # background
    BG_BLACK = "\u001b[40m"
    BG_RED = "\u001b[41m"
    BG_GREEN = "\u001b[42m"
    BG_YELLOW = "\u001b[43m"
    BG_BLUE = "\u001b[44m"
    BG_MAGENTA = "\u001b[45m"
    BG_CYAN = "\u001b[46m"
    BG_WHITE = "\u001b[47m"

    # background, bright version
    BG_BLACK_BR = "\u001b[40;1m"
    BG_RED_BR = "\u001b[41;1m"
    BG_GREEN_BR = "\u001b[42;1m"
    BG_YELLOW_BR = "\u001b[43;1m"
    BG_BLUE_BR = "\u001b[44;1m"
    BG_MAGENTA_BR = "\u001b[45;1m"
    BG_CYAN_BR = "\u001b[46;1m"
    BG_WHITE_BR = "\u001b[47;1m"


def file_message(filename):
    print("--> {dec1}{dec2}{filename}{reset}".format(
        dec1=PrintDecorations.UNDERLINE,
        dec2=PrintDecorations.MAGENTA_BR,
        filename=filename,
        reset=PrintDecorations.RESET))


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        file_message(dir)


def files(dir=None, ext=None, in_dir=None, out_dir=None, in_ext=None, out_ext=None):
    if dir is None and ext is None:
        if not os.path.exists(in_dir):
            return []
        filenames = [f for f in os.listdir(in_dir) if os.path.splitext(f)[1].lower() == in_ext]
        in_files = [in_dir + "/" + f for f in filenames]
        out_files = [out_dir + "/" + os.path.splitext(f)[0] + out_ext for f in filenames]
        res = zip(in_files, out_files)
    else:
        if not os.path.exists(dir):
            return []
        filenames = [dir + "/" + f for f in os.listdir(dir) if os.path.splitext(f)[1].lower() == ext]
        res = filenames
    return res

