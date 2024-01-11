from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize

def calculate_metrics(original_path, compared_path):
    try:
        # Load images
        original = io.imread(original_path)
        compared = io.imread(compared_path)
    except Exception as e:
        print(f"Error loading images: {e}")
        return None, None

    # Convert images to float
    original = img_as_float(original)
    min_dim = min(original.shape[0], original.shape[1])
    win_size = min_dim if min_dim % 2 == 1 else min_dim - 1

    # Resize images to the minimum common size
    min_height = min(original.shape[0], compared.shape[0])
    min_width = min(original.shape[1], compared.shape[1])

    original = resize(original, (min_height, min_width), anti_aliasing=True)
    compared = resize(compared, (min_height, min_width), anti_aliasing=True)

    # Calculate metrics
    psnr = peak_signal_noise_ratio(original, compared)
    ssim = structural_similarity(original, compared, multichannel=True, win_size=win_size)

    return psnr, ssim

# Use the function
psnr, ssim = calculate_metrics("test.jpg", "output/UnBlurApp_20231123103626.png")

if psnr is not None and ssim is not None:
    print(f"PSNR: {psnr}, SSIM: {ssim}")
else:
    print("Error calculating metrics.")