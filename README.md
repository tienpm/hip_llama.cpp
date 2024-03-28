# hip_llama.cpp

Inference llama2 model on the AMD GPUs system

## Getting Started

- Install the dependencies.
- Clone the repository.
- Run the project.

```
cd hip_llama.cpp
make
```

## Usage

- Instructions on how to use the project.

```bash
./build/apps/llama model.bin -m test -f <input_filename> -o <output_filename>
```

- Examples of how to inference llama2.

```bash
./build/apps/llama /shared/erc/getpTA/main/modelbin/stories110M.bin -m test -f assets/in/gen_in_128.txt -o assets/out/gen_out_128.txt
```

## Documentation

- Not available yet.

## Contributing

- If you have some issues or feature requests, please open issues or let us know by email of contributors bellow.

## License

[GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html)

## Contributers

| Full Name        | Email                     |
| ---------------- | ------------------------- |
| Pham Manh Tien   | tienpmuit@gmail.com       |
| Nguyen Huy Hoang | hoang.nguyen@moreh.com.vn |
| Nguyen Xuan Anh  | xuananhbka@gmail.com      |

## Acknowledgments

Reference:

- [llama2.c](https://github.com/karpathy/llama2.c)
- [llama2.c for Dummies](https://github.com/RahulSChand/llama2.c-for-dummies?tab=readme-ov-file)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
