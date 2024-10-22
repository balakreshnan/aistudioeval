import qrcode

def main():
    # Data for the QR code (this could be a URL, text, etc.)
    #data = "https://microsoft.benevity.org/campaigns/82923"
    data = "https://www.amazon.com/dp/B0DK7S55BB"

    # Create a QR code instance
    qr = qrcode.QRCode(
        version=1,  # Version controls the size of the QR Code (1 is the smallest)
        error_correction=qrcode.constants.ERROR_CORRECT_L,  # Controls error correction level
        box_size=10,  # Size of each box in the QR code grid
        border=4,  # Thickness of the border (4 is the minimum)
    )

    # Add data to the instance
    qr.add_data(data)
    qr.make(fit=True)  # Fit the data to the size of the QR code

    # Create an image from the QR code instance
    img = qr.make_image(fill="black", back_color="white")

    # Save the image
    img.save("aiforfoodbook2024.png")

    print("QR code generated and saved as 'aiforfoodbook2024.png'")

if __name__ == "__main__":
    main()