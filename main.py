from game import Game


def main():
    g = Game()
    print("玩法: 输入 `a b`(1~6) 把 a 列栈顶整段移到 b 列; 3 个相同合成 n+1; 9 消失得分; 列高超 7 即败。")
    print("节奏: 开局每列 2 个 1; 每轮第 2 次移动后显示预告, 第 3 次移动后按预告掉落(预告=实际掉落)。")
    print("命令: q 退出, n 新局")
    while True:
        print(g.render())
        if g.dead:
            print("游戏结束！得分:", g.score)
            raw = input("再来一局? [y/n]: ").strip().lower()
            if raw == "y":
                g = Game()
                continue
            break
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if raw in ("q", "quit", "exit"):
            break
        if raw in ("n", "new"):
            g = Game()
            continue
        parts = raw.split()
        if len(parts) != 2:
            print("格式: a b")
            continue
        try:
            src, dst = int(parts[0]) - 1, int(parts[1]) - 1
        except ValueError:
            print("请输入数字")
            continue
        if not g.move(src, dst):
            print("非法操作: 源列为空或同列")


if __name__ == "__main__":
    main()
