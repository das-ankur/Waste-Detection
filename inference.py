def infer(model, image_path):
    results = model(image_path)
    results.save()
    return results.pandas().xyxy[0].to_dict()


