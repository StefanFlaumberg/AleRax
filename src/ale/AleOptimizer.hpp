#pragma once

#include "AleState.hpp"
#include <search/SpeciesRootSearch.hpp>
#include <trees/SpeciesTree.hpp>
#include <IO/FamiliesFileParser.hpp>
#include "UndatedDLMultiModel.hpp"
#include "UndatedDTLMultiModel.hpp"
#include <trees/PLLRootedTree.hpp>
#include <parallelization/PerCoreGeneTrees.hpp>
#include <memory>
#include <vector>
#include <maths/ModelParameters.hpp>
#include "AleEvaluator.hpp"

struct ScoredHighway {
  ScoredHighway() 
  {}

  
  ScoredHighway(const Highway &highway, double score = 0.0, double scoreDiff = 0.0):
    highway(highway),
    score(score),
    scoreDiff(scoreDiff)
  {}

  Highway highway;
  double score;
  double scoreDiff;
  bool operator < (const ScoredHighway &other) const {
    return score < other.score;
  }
};

bool cmpHighwayByProbability(const ScoredHighway &a, const ScoredHighway &b);


class AleOptimizer: public SpeciesTree::Listener {
public:
  AleOptimizer(const std::string speciesTreeFile, 
      const Families &families, 
      const RecModelInfo &info,
      const Parameters &startingRates,
      bool optimizeRates,
      bool optimizeVerbose,
      const std::string &speciesCategoryFile,
      const std::string &outputDir);

  /**
   *  Optimize the species tree topology
   */
  void optimize();

  /**
   *  Optize the species tree root
   */
  double rootSearch(unsigned int maxDepth, bool thorough = false);
 
  /**
   *  Callback called when the species tree topology changes
   */
  void onSpeciesTreeChange(const std::unordered_set<corax_rnode_t *> *nodesToInvalidate);

  /**
   *  Sample reconciliations and generate many output files
   */
  void reconcile(unsigned int samples);

  /**
   *  Optimize the model parameters
   */
  double optimizeModelRates(bool thorough = false);

  /**
   *  Optimize the relative order of speciation events
   */
  void optimizeDates(bool thorough = true);


  /**
   *  Randomly pick a branch in the species tree and reroot it
   *  at this branch
   */
  void randomizeRoot();

  /**
   *  Save the species tree
   */
  void saveSpeciesTree();

  /**
   *  Save the species tree with its support values
   */
  void saveSupportTree();

  /**
   *  Save the DTL rates and the per-family likelihoods
   */
  void saveRatesAndLL();

  /**
   *  Accessor
   */
  AleEvaluator &getEvaluator() {return *_evaluator;}
  
  /**
   *  Accessor
   */
  SpeciesTree &getSpeciesTree() {return *_state.speciesTree;}
 
  /**
   *  Accessor
   */
  AleModelParameters &getModelParameters() {return _state.modelParameters;}
  const AleModelParameters &getModelParameters() const {return _state.modelParameters;}

  const RecModelInfo &getRecModelInfo() const {return _info;}

  void saveBestHighways(const std::vector<ScoredHighway> &highways,
      const std::string &output);
  void saveRELLSupports();
  std::string getHighwaysOutputDir() const;
private:
  AleState _state;
  const Families &_families;
  PerCoreGeneTrees _geneTrees;
  RecModelInfo _info;
  std::unique_ptr<AleEvaluator> _evaluator;
  std::string _outputDir;
  SpeciesSearchState _speciesTreeSearchState;
  RootLikelihoods _rootLikelihoods;
  
  double sprSearch(unsigned int radius);
  double transferSearch();
  std::string saveCurrentSpeciesTreeId(std::string str = "inferred_species_tree.newick", bool masterRankOnly = true);
  void saveCurrentSpeciesTreePath(const std::string &str, bool masterRankOnly = true);
};



