import pkuseg


if __name__ == "__main__":
    # seg = pkuseg.PKUSeg()

    # print(seg.cut("我爱北京天安门。"))

    # pkuseg.test("test.txt", "temp.txt", verbose=True)

    pkuseg.train(
        "data/pku_test_gold.utf8", "data/pku_test_gold.utf8", "models/"
    )

