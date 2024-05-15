def program():
    mat_str = """
    0,2,1,0,0,0,2,0,0,0,28717;
    0,1,0,2,0,0,1,0,0,0,2612;
    0,3,2,0,0,1,0,0,0,0,2371;
    0,0,3,0,1,0,1,0,0,0,11957;
    0,1,0,3,1,0,0,0,0,0,8368;
    3,0,0,0,0,0,1,1,0,0,9978;
    2,1,0,0,0,0,0,0,1,0,9203;
    0,2,1,1,0,0,0,0,0,0,11275;
    0,0,0,0,1,0,0,1,0,0,9257;
    0,5,0,1,0,0,0,0,0,0,30226;
    0,1,2,0,1,0,0,0,1,0,4885;
    1,0,0,0,0,0,1,0,1,0,22566;
    0,1,1,2,0,0,0,1,0,0,2468;
    0,0,2,0,1,0,0,0,1,0,11513;
    2,2,2,0,0,1,0,0,0,0,23705;
    5,0,2,0,0,0,0,1,0,0,11874;
    2,1,1,0,1,0,0,0,1,0,32110;
    0,0,0,1,0,1,2,0,0,0,7255;
    2,1,0,1,0,1,1,0,0,0,1245;
    1,0,2,2,0,0,0,0,0,0,3783;
    1,3,1,1,0,0,0,0,0,0,12000;
    0,1,0,0,0,0,1,1,0,0,7607;
    0,3,2,0,0,0,0,0,0,1,8637;
    """
    mat = Matrix(mat_str)
    arr = list(mat.get_arr.tolist())
    print([len(row) for row in arr])
    while True:
        sub_arr = list(sample(arr, 10))

        b = []
        for i in range(len(sub_arr)):
            b.append(sub_arr[i].pop())

        print(sub_arr)
        mat_arr = np.array(sub_arr)

        mat = Matrix(mat_arr)
        if mat.get_determinant != 0:
            print(len(mat.get_arr))
            break

        for i in range(len(sub_arr)):
            sub_arr[i].append(b[i])
    print(mat)
    print(b)

    mat.append_b(np.asfarray(b))
    sol = mat.solve_mod_p(17)
    print(sol)
    print(mat)

    x = [0, 5, 37, sol[1]]
    n = [4, 9, 43, 17]
    print(crt(n, x))

def program():

    p = np.poly1d([-1, 7, -7])
    q = np.poly1d([3, -3, 8])
    x = np.arange(20)
    y = p(x)
    y_2 = q(x)
    plt.plot(x, y, label="p")
    plt.plot(x, y_2, label="q")
    plt.legend()
    plt.grid()
    plt.show()