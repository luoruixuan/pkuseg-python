import pkuseg
import cProfile
import pstats

if __name__ == "__main__":
    seg = pkuseg.PKUSeg()

    print(seg.cut("我爱北京天安门。"))

    # pkuseg.test("data/pku_test.utf8", "temp.txt", verbose=True, nthread=1)

    cProfile.run(
        'pkuseg.test("data/pku_test.utf8", "temp.txt", verbose=True, nthread=1)',
        "cfstats",
    )
    p = pstats.Stats("cfstats")
    p.sort_stats("cumulative").print_stats()
    # pkuseg.test(TEST_FILE, "output1.txt")

    # pkuseg.train(
    #     "data/pku_test_gold.utf8", "data/pku_test_gold.utf8", "models/"
    # )

