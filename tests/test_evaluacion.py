"""Comprueba que el test no se consulte antes de cerrar la selección."""

import contextlib
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluar_test


class EvaluacionTest(unittest.TestCase):
    def test_reutiliza_resultados_sin_cargar_modelo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "docs/evaluacion"
            output.mkdir(parents=True)
            saved = {"modelo": "regresion_logistica", "f1_macro": 0.931}
            path = output / "metricas.json"
            path.write_text(json.dumps(saved), encoding="utf-8")
            before = path.read_bytes()
            with patch.object(evaluar_test, "ROOT", root), patch.object(sys, "argv", ["evaluar_test.py"]):
                with patch.object(evaluar_test, "cargar_resultados") as load_results:
                    with patch.object(evaluar_test.joblib, "load") as load_model:
                        with contextlib.redirect_stdout(StringIO()):
                            evaluar_test.main()
                        load_results.assert_not_called()
                        load_model.assert_not_called()
            self.assertEqual(before, path.read_bytes())

    def test_rechaza_seleccion_incompleta_antes_de_leer_test(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(evaluar_test, "ROOT", Path(directory)):
                with patch.object(sys, "argv", ["evaluar_test.py"]):
                    with patch.object(evaluar_test, "cargar_resultados", return_value=pd.DataFrame(index=["baseline"])):
                        with patch.object(evaluar_test.pd, "read_csv") as read_csv:
                            with self.assertRaisesRegex(ValueError, "Faltan modelos"):
                                evaluar_test.main()
                            read_csv.assert_not_called()


if __name__ == "__main__":
    unittest.main()
