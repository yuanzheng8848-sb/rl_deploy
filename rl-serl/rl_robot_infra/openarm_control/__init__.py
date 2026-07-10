"""OpenArm control package.

Keep package import lightweight. Import concrete classes from their modules,
e.g. openarm_control.controller.OpenArmController, so mock/test paths can use
openarm_control.types without importing the native openarm_can extension.
"""
