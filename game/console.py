from .env import Game, NUM_COLS, HEIGHT


def render(g: Game):
    print()
    head = f"步数 {g.step:3d}   得分 {g.score:5d}   合成上限 {g.max_created}"
    if g.preview:
        head += "   预告: " + " ".join(map(str, g.preview))
    print(head)
    print("     (0)    ", end="")
    print("    ".join(f"({i})" for i in range(1, NUM_COLS)))
    for r in range(HEIGHT):
        cells = []
        for c in range(NUM_COLS):
            st = g.stacks[c]
            idx = len(st) - HEIGHT + r
            cells.append("| " + (str(st[idx]) if 0 <= idx < len(st) else " ") + " |")
        label = "顶" if r == 0 else ("底" if r == HEIGHT - 1 else "  ")
        print(f"{label}  " + "  ".join(cells))
    for m in g.messages:
        print("   " + m)
