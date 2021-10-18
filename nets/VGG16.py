from torchvision.models import vgg16

vgg16_model = vgg16()

# for child in vgg16_model.children():
#     print(child)

object_methods = [method_name for method_name in dir(vgg16_model)
                  if callable(getattr(vgg16_model, method_name))]

# print(object_methods)
VGG16 = vgg16_model


# vgg16_model.classifier = vgg16_model.classifier[:-1]
# print(vgg16_model.classifier)
