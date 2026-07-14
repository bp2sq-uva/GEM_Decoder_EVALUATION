// inference_cpp.C
//
// Usage:
// root -l -b -q 'inference_cpp.C("filtered_output.root","hit_centers.txt","sbs.gemFT.m0")'
//
// This reads only regular ADC hit/cluster information.
// It does NOT read any goodADC branches.

#include <TChain.h>
#include <TTree.h>
#include <TBranch.h>
#include <TString.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

double ClusterCenterStrip(double lo, double nstrips){
  return lo + 0.5 * (nstrips - 1.0);
}

void inference_cpp(
    const char *infile = "/volatile/halla/sbs/bhasitha/Tracking_ML/GEM_Decoder_EVALUATION/filtered_replayed_withoutROIcut.root",
    const char *txtoutfile = "Scratch/CPP/textfile_outputs/hit_centers_CPP_withoutROIcut.txt",
    const TString prefix = "sbs.gemFT.m0"
){
  TChain *T = new TChain("T");
  T->Add(infile);

  if(!T || T->GetEntries() == 0){
    std::cerr << "ERROR: Could not find tree T, or tree is empty.\n";
    return;
  }

  TString p = prefix;

  constexpr int NMAXHITS = 200000;

  double nhit = 0.0;

  std::vector<double> hit_iu(NMAXHITS, 0.0);
  std::vector<double> hit_iv(NMAXHITS, 0.0);

  std::vector<double> clu_lo(NMAXHITS, 0.0);
  std::vector<double> clu_n(NMAXHITS, 0.0);

  std::vector<double> clv_lo(NMAXHITS, 0.0);
  std::vector<double> clv_n(NMAXHITS, 0.0);

  // Disable everything first so ROOT does not read unnecessary branches.
  T->SetBranchStatus("*", 0);

  // Enable only regular ADC hit and cluster branches.
  T->SetBranchStatus(p + ".hit.nhits2d", 1);
  T->SetBranchStatus(p + ".hit.hit_iuclust", 1);
  T->SetBranchStatus(p + ".hit.hit_ivclust", 1);

  T->SetBranchStatus(p + ".clust.clustu_istriplo", 1);
  T->SetBranchStatus(p + ".clust.clustu_strips", 1);

  T->SetBranchStatus(p + ".clust.clustv_istriplo", 1);
  T->SetBranchStatus(p + ".clust.clustv_strips", 1);

  // Set branch addresses.
  T->SetBranchAddress(p + ".hit.nhits2d", &nhit);
  T->SetBranchAddress(p + ".hit.hit_iuclust", hit_iu.data());
  T->SetBranchAddress(p + ".hit.hit_ivclust", hit_iv.data());

  T->SetBranchAddress(p + ".clust.clustu_istriplo", clu_lo.data());
  T->SetBranchAddress(p + ".clust.clustu_strips", clu_n.data());

  T->SetBranchAddress(p + ".clust.clustv_istriplo", clv_lo.data());
  T->SetBranchAddress(p + ".clust.clustv_strips", clv_n.data());

  TBranch *b_nhit = T->GetBranch(p + ".hit.nhits2d");

  if(!b_nhit){
    std::cerr << "ERROR: Missing required branch: "
              << p << ".hit.nhits2d\n";
    std::cerr << "Check the prefix. Current prefix = " << p << "\n";
    return;
  }

  std::ofstream out(txtoutfile);

  if(!out.is_open()){
    std::cerr << "ERROR: Could not open output text file: "
              << txtoutfile << "\n";
    return;
  }

  out << "Event_ID"
      << "\t" << "2D_hit_ID"
      << "\t" << "Hit_center_U_strip_ID"
      << "\t" << "Hit_center_V_strip_ID"
      << "\n";

  Long64_t nentries = T->GetEntries();

  Long64_t nWritten = 0;
  Long64_t nSkippedTooManyHits = 0;
  Long64_t nSkippedBadClusterIndex = 0;

  for(Long64_t ev = 0; ev < nentries; ev++){

    // Read only nhits first, so oversized events can be skipped safely.
    b_nhit->GetEntry(ev);

    int Nh = int(nhit);

    if(Nh < 0){
      std::cerr << "WARNING: Event " << ev
                << " has negative nhits2d = " << Nh
                << ". Skipping.\n";
      continue;
    }

    if(Nh > NMAXHITS){
      nSkippedTooManyHits++;
      std::cerr << "WARNING: Event " << ev
                << " has too many regular ADC 2D hits: "
                << Nh << ". NMAXHITS = " << NMAXHITS
                << ". Skipping.\n";
      continue;
    }

    T->GetEntry(ev);

    for(int ih = 0; ih < Nh; ih++){

      int uidx = int(hit_iu[ih]);
      int vidx = int(hit_iv[ih]);

      if(uidx < 0 || vidx < 0 ||
         uidx >= NMAXHITS || vidx >= NMAXHITS){
        nSkippedBadClusterIndex++;
        continue;
      }

      double ucenter = ClusterCenterStrip(clu_lo[uidx], clu_n[uidx]);
      double vcenter = ClusterCenterStrip(clv_lo[vidx], clv_n[vidx]);

      out << ev
          << "\t" << ih
          << "\t" << ucenter
          << "\t" << vcenter
          << "\n";

      nWritten++;
    }
  }

  out.close();

  std::cout << "Wrote text file: " << txtoutfile << "\n";
  std::cout << "Rows written: " << nWritten << "\n";
  std::cout << "Events skipped due to too many regular ADC hits: "
            << nSkippedTooManyHits << "\n";
  std::cout << "Hits skipped due to bad U/V cluster index: "
            << nSkippedBadClusterIndex << "\n";
}