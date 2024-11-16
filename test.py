import unittest
from ML_FLOW2 import ML_FLOW_PARCIAL2

class prueba(unittest.TestCase):
    def test_model(self):
        modelo = ML_FLOW_PARCIAL2()
        salida = modelo.ML_FLOW()

        if salida["success"]==True:
            self.assertTrue(salida["success"],"Completed succesfully ...")
            a = "Completed succesfully ..."
            b = salida["message"]
            print({'Process': a, 'Message': b})
        else:
            a = "Not completed succesfully ..."
            b = salida["message"]
            print({'Process': a, 'Message': b})

pb = prueba()
print(pb.test_model())

if __name__ == "_main_":
    unittest.main()