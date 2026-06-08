import os
from typing import Any, Dict
from src.config_helper import load_settings
from src.ingestion.crawler import CrawlerAgent
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.preprocessor import Preprocessor
from src.rag.embedder import Embedder
from src.rag.vectorstore import VectorStore
from src.rag.retriever import Retriever
from src.inference.generator import Generator
from src.evaluation.judge_agent import JudgeAgent
from src.evaluation.scoring import ScoringSystem
from src.training.dataset_builder import DatasetBuilder
from src.training.trainer import QLoraTrainer
from src.training.orchestrator import TrainingOrchestrator

class AutonomousPipeline:
    def __init__(self, config_path: str = "config/settings.yaml", settings: Any = None):
        if settings is not None:
            self.settings = settings
        else:
            self.settings = load_settings(config_path)

        
        # 1. Ingestion Components
        self.crawler = CrawlerAgent(self.settings)
        self.parser = PDFParser(self.settings)
        self.preprocessor = Preprocessor(self.settings)
        
        # 2. RAG Components
        self.embedder = Embedder(self.settings)
        self.vectorstore = VectorStore(self.settings)
        self.retriever = Retriever(self.vectorstore, self.embedder, self.settings)
        
        # 3. Gen & Eval Components
        self.generator = Generator(self.settings)
        self.judge = JudgeAgent(self.settings)
        self.scoring = ScoringSystem(self.settings)
        
        # 4. Training Components
        self.dataset_builder = DatasetBuilder(self.settings)
        self.trainer = QLoraTrainer(self.settings)
        self.orchestrator = TrainingOrchestrator(self.settings, self.trainer)

    def run_single_drug_pipeline(self, drug: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """Runs the entire generation, evaluation, and conditional learning loop for a single drug."""
        code = drug["japic_code"]
        name = drug.get("name_ja", "Unknown")
        print(f"\n=== [Pipeline] Starting iteration {iteration} for drug: {name} ({code}) ===")

        # Step 1: Download raw documents
        raw_results = self.crawler.collect([drug])
        drug_data = next((d for d in raw_results if d["japic_code"] == code), None)
        if not drug_data:
            raise ValueError(f"Failed to crawl/locate drug data for code: {code}")

        # Step 2: PDF Parsing to Markdown (CTD)
        ctd_md = ""
        if drug_data.get("ctd_pdf"):
            ctd_md = self.parser.parse(drug_data["ctd_pdf"])
        else:
            # Fallback mock for testing pipeline flow
            ctd_md = self.parser._run_basic_fallback("mock_ctd.pdf")

        # Step 3: Extract ground truths
        gt_data = self.preprocessor.extract_targets(drug_data)
        true_if = gt_data["true_if"]
        true_siori = gt_data["true_siori"]

        # Step 4: Index CTD chunks into RAG
        if ctd_md:
            self.vectorstore.index_document(ctd_md, code, self.embedder)

        # Step 5: Inference (Generate IF and Siori)
        generated_if = self.generator.generate(self.retriever, "interview_form", drug)
        generated_siori = self.generator.generate(self.retriever, "siori", drug)

        # Step 6: Evaluate generated IF (we score IF as primary target, Siori as secondary)
        report = self.judge.evaluate(generated_if, true_if)
        
        # Log evaluation metrics to persistent score history
        self.scoring.log_score(code, iteration, report)

        # Step 7: Conditional Auto-Training (Orchestrated feedback loop)
        target_score = self.settings.pipeline.target_score
        current_score = report.get("total_score", 0)
        
        if current_score < target_score:
            # Append failed example to training dataset
            self.dataset_builder.append_and_update(ctd_md, true_if, true_siori, report)
            
            # Trigger fine-tuning if score below target
            new_adapter = self.orchestrator.handle_feedback(
                score_report=report,
                dataset_path=self.dataset_builder.dataset_file,
                iteration=iteration
            )
            
            # Reload updated model adapter weights in generator
            if new_adapter:
                self.generator.reload_adapter(new_adapter)
        else:
            print(f"[Pipeline] Goal achieved for {name}! Score: {current_score} >= {target_score}")
            
        # Save generated outputs to files
        gen_dir = os.path.join(self.settings.paths.outputs_generated, f"JapicCode_{code}")
        os.makedirs(gen_dir, exist_ok=True)
        with open(os.path.join(gen_dir, "generated_if.md"), "w", encoding="utf-8") as f:
            f.write(generated_if)
        with open(os.path.join(gen_dir, "generated_siori.md"), "w", encoding="utf-8") as f:
            f.write(generated_siori)
            
        return {
            "score": current_score,
            "report": report,
            "generated_if": generated_if,
            "generated_siori": generated_siori
        }
        
    def run_all(self):
        """Runs the loop sequentially through the configured drug list."""
        for drug in self.crawler.static_drugs:
            # Iteratively refine model on the drug until goal score met or max iterations reached
            iteration = 1
            max_iters = self.settings.pipeline.max_iterations
            while iteration <= max_iters:
                res = self.run_single_drug_pipeline(drug, iteration)
                if res["score"] >= self.settings.pipeline.target_score:
                    break
                iteration += 1
