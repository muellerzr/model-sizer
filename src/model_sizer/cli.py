import argparse
import prettytable
from .utils import get_sizes, create_empty_model, convert_bytes

def main():
    parser = argparse.ArgumentParser(description="Model Sizer")
    parser.add_argument("--model_name", type=str, help="The model name on the Hugging Face Hub")
    parser.add_argument("--library_name", type=str, help="The library the model has an integration with, such as `transformers`, needed only if this information is not stored on the Hub.")
    parser.add_argument(
        "--dtypes", 
        type=str, 
        nargs="+", 
        default=["float32"], 
        help="The dtypes to use for the model, must be one (or many) of `float32`, `float16`, `int8`, and `int4`", 
        choices=["float32", "float16", "int8", "int4"]
    )

    args = parser.parse_args()

    model = create_empty_model(args.model_name, library_name=args.library_name)
    total_size, largest_layer = get_sizes(model)

    table = prettytable.PrettyTable()
    table.title = f"Memory Usage for `{args.model_name}`"

    table.field_names = ["dtype", "Largest Layer", "Total Size", "Training using Adam"]
    for dtype in args.dtypes:
        dtype_total_size = total_size
        dtype_largest_layer = largest_layer[0]
        if dtype == "float16":
            dtype_total_size /= 2
            dtype_largest_layer /= 2
        elif dtype == "int8":
            dtype_total_size /= 4
            dtype_largest_layer /= 4
        elif dtype == "int4":
            dtype_total_size /= 8
            dtype_largest_layer /= 8
        dtype_training_size = convert_bytes(dtype_total_size * 4)
        dtype_total_size = convert_bytes(dtype_total_size)
        dtype_largest_layer = convert_bytes(dtype_largest_layer)
        table.add_row([dtype, dtype_largest_layer, dtype_total_size, dtype_training_size])
    print(table)