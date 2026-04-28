"""
Guardrails de segurança para input e output do agente.
"""
import re

class InputGuardrail:
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"you\s+are\s+now\s+a",
        r"system:\s*",
        r"<\|im_start\|>",
        r"\[INST\]",
        r"forget\s+(everything|all|your\s+instructions)",
        r"act\s+as\s+(a\s+)?",
        r"pretend\s+you\s+are",
    ]

    def __init__(self, max_length: int = 4096) -> None:
        self.max_length = max_length
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]

    def validate(self, user_input: str) -> tuple[bool, str]:
        """Valida input do usuário.

        Args:
            user_input: Texto do usuário.
        """
        for pattern in self._compiled:
            if pattern.search(user_input):
                return False, "Input bloqueado: padrão suspeito detectado."

        if len(user_input) > self.max_length:
            return False, f"Input bloqueado: excede tamanho máximo ({self.max_length} chars)."

        return True, "OK"


class OutputGuardrail:
    """Valida e sanitiza output do LLM antes de retornar ao usuário."""

    def __init__(self, language: str = "pt") -> None:
        self.language = language
        self._analyzer = None
        self._anonymizer = None

    def _ensure_loaded(self) -> None:
        if self._analyzer is None:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()

    def sanitize(self, llm_output: str) -> str:
        """Remove PII do output do LLM.

        Args:
            llm_output: Texto gerado pelo LLM.
        """
        self._ensure_loaded()

        results = self._analyzer.analyze(
            text=llm_output,
            language=self.language,
            entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "BR_CPF"],
        )

        if results:
            logger.warning("PII detectado no output: %d entidades", len(results))
            anonymized = self._anonymizer.anonymize(
                text=llm_output,
                analyzer_results=results,
            )
            return anonymized.text

        return llm_output