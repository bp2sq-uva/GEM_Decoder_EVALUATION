// filter_goodADC_events.C
//
// Usage:
// root -l 'filter_goodADC_events.C("/path/to/replayed_*.root","filtered_replayed.root")'
//
// This macro copies the original T tree to a new ROOT file,
// keeping the same branches and same event data,
// but only for events that pass the filtering checks.
//
// Important:
// This version rejects the whole event if even one positive goodADC strip
// lies outside the ROI strip range.

#include <TFile.h>
#include <TTree.h>
#include <TChain.h>
#include <TString.h>
#include <TObject.h>

#include <iostream>

void filter_goodADC_events(
    const char *input_pattern,
    const char *outfile = "filtered_replayed.root",
    const TString prefix = "sbs.gemFT.m0"
){
  constexpr int NMAXHITS   = 1000;
  constexpr int NMAXSTRIPS = 20000;

  // ----------------------------------------------------
  // Use TChain so wildcard input works.
  // ----------------------------------------------------

  TChain *T = new TChain("T");

  int nfiles = T->Add(input_pattern);

  if(nfiles <= 0 || T->GetEntries() == 0){
    std::cerr << "Error: no input files/events found for pattern:\n"
              << input_pattern << std::endl;
    delete T;
    return;
  }

  std::cout << "Input pattern: " << input_pattern << std::endl;
  std::cout << "Files added:   " << nfiles << std::endl;
  std::cout << "Total events:  " << T->GetEntries() << std::endl;

  // ----------------------------------------------------
  // Variables needed only for filtering.
  // ----------------------------------------------------

  double nhit = 0.;
  double ngood = 0.;

  double nstrip = 0.;
  double nstrip_goodADC = 0.;

  double strip_istrip[NMAXSTRIPS] = {0.};
  double strip_IsU[NMAXSTRIPS] = {0.};
  double strip_IsV[NMAXSTRIPS] = {0.};
  double strip_ADCsum[NMAXSTRIPS] = {0.};

  double good_strip_istrip[NMAXSTRIPS] = {0.};
  double good_strip_IsU[NMAXSTRIPS] = {0.};
  double good_strip_IsV[NMAXSTRIPS] = {0.};
  double good_strip_ADCsum[NMAXSTRIPS] = {0.};

  double roi_inmod = 0.;
  double roi_ustrip_min = 0.;
  double roi_ustrip_max = 0.;
  double roi_vstrip_min = 0.;
  double roi_vstrip_max = 0.;

  TString p = prefix;

  // ----------------------------------------------------
  // Set branch addresses for filtering.
  // ----------------------------------------------------

  T->SetBranchAddress(p + ".hit.nhits2d", &nhit);
  T->SetBranchAddress(p + ".goodADChit.nhits2d", &ngood);

  T->SetBranchAddress(p + ".strip.nstripsfired", &nstrip);
  T->SetBranchAddress(p + ".strip.istrip", strip_istrip);
  T->SetBranchAddress(p + ".strip.IsU", strip_IsU);
  T->SetBranchAddress(p + ".strip.IsV", strip_IsV);
  T->SetBranchAddress(p + ".strip.ADCsum", strip_ADCsum);

  T->SetBranchAddress(p + ".strip.nstripsfired_goodADC", &nstrip_goodADC);
  T->SetBranchAddress(p + ".strip.istrip_goodADC", good_strip_istrip);
  T->SetBranchAddress(p + ".strip.IsU_goodADC", good_strip_IsU);
  T->SetBranchAddress(p + ".strip.IsV_goodADC", good_strip_IsV);
  T->SetBranchAddress(p + ".strip.ADCsum_goodADC", good_strip_ADCsum);

  T->SetBranchAddress(p + ".roi.inmod", &roi_inmod);
  T->SetBranchAddress(p + ".roi.ustrip_min", &roi_ustrip_min);
  T->SetBranchAddress(p + ".roi.ustrip_max", &roi_ustrip_max);
  T->SetBranchAddress(p + ".roi.vstrip_min", &roi_vstrip_min);
  T->SetBranchAddress(p + ".roi.vstrip_max", &roi_vstrip_max);

  // ----------------------------------------------------
  // Create output file and clone the full tree structure.
  // CloneTree(0) keeps all branches but starts with zero events.
  // ----------------------------------------------------

  TFile *fout = TFile::Open(outfile, "RECREATE");

  if(!fout || fout->IsZombie()){
    std::cerr << "Error: cannot create output file " << outfile << std::endl;
    delete T;
    return;
  }

  fout->cd();

  TTree *Tout = T->CloneTree(0);

  // ----------------------------------------------------
  // Counters.
  // ----------------------------------------------------

  Long64_t nentries = T->GetEntries();

  Long64_t nPassed = 0;
  Long64_t nSkippedTooManyHits = 0;
  Long64_t nSkippedNoROI = 0;
  Long64_t nSkippedTooManyStrips = 0;
  Long64_t nSkippedBadStripMatch = 0;
  Long64_t nSkippedGoodADCStripOutsideROI = 0;
  Long64_t nSkippedOver160 = 0;

  // ----------------------------------------------------
  // Event loop.
  // ----------------------------------------------------

  for(Long64_t ev = 0; ev < nentries; ev++){

    T->GetEntry(ev);

    bool pass_event = true;

    int Nh  = int(nhit);
    int Ng  = int(ngood);
    int Ns  = int(nstrip);
    int Nsg = int(nstrip_goodADC);

    // ----------------------------------------------------
    // Filter 1:
    // Skip event if hit arrays are larger than allowed.
    // ----------------------------------------------------

    if(Nh > NMAXHITS || Ng > NMAXHITS){
      nSkippedTooManyHits++;
      pass_event = false;
    }

    // ----------------------------------------------------
    // Filter 2:
    // Skip event if ROI does not overlap this module.
    // ----------------------------------------------------

    if(pass_event && roi_inmod == 0.){
      nSkippedNoROI++;
      pass_event = false;
    }

    // ----------------------------------------------------
    // Filter 3:
    // Skip event if strip arrays are larger than allowed.
    // ----------------------------------------------------

    if(pass_event && (Ns > NMAXSTRIPS || Nsg > NMAXSTRIPS)){
      nSkippedTooManyStrips++;
      pass_event = false;
    }

    // ----------------------------------------------------
    // Filter 4:
    // Reject event if ANY positive goodADC strip is outside ROI.
    //
    // For U strips, require:
    //   roi_ustrip_min < strip < roi_ustrip_max
    //
    // For V strips, require:
    //   roi_vstrip_min < strip < roi_vstrip_max
    //
    // This uses strict boundaries, matching the style used in
    // the original ROI strip counting logic.
    // ----------------------------------------------------

    if(pass_event){

      bool all_goodADC_strips_inside_roi = true;

      int ru_min = int(roi_ustrip_min);
      int ru_max = int(roi_ustrip_max);
      int rv_min = int(roi_vstrip_min);
      int rv_max = int(roi_vstrip_max);

      for(int igs = 0; igs < Nsg; igs++){

        if(good_strip_ADCsum[igs] <= 0.0) continue;

        double gstrip = good_strip_istrip[igs];

        if(good_strip_IsU[igs]){
          if(gstrip <= ru_min || gstrip >= ru_max){
            all_goodADC_strips_inside_roi = false;
            break;
          }
        }
        else if(good_strip_IsV[igs]){
          if(gstrip <= rv_min || gstrip >= rv_max){
            all_goodADC_strips_inside_roi = false;
            break;
          }
        }
      }

      if(!all_goodADC_strips_inside_roi){
        nSkippedGoodADCStripOutsideROI++;
        pass_event = false;
      }
    }

    // ----------------------------------------------------
    // Filter 5:
    // Require at least one goodADC U strip and one goodADC V strip
    // to have a matching regular strip with ADCsum > 0.
    //
    // Matching means:
    //   same IsU / IsV
    //   same strip number
    //   regular strip ADCsum > 0
    // ----------------------------------------------------

    if(pass_event){

      int matching_goodADC_stripsU = 0;
      int matching_goodADC_stripsV = 0;

      for(int igs = 0; igs < Nsg; igs++){

        if(good_strip_ADCsum[igs] <= 0.0) continue;

        double gstrip = good_strip_istrip[igs];
        double gIsU   = good_strip_IsU[igs];
        double gIsV   = good_strip_IsV[igs];

        for(int irs = 0; irs < Ns; irs++){

          bool same_axis =
            (strip_IsU[irs] == gIsU) &&
            (strip_IsV[irs] == gIsV);

          bool same_strip =
            (strip_istrip[irs] == gstrip);

          if(!same_axis || !same_strip) continue;

          if(strip_ADCsum[irs] > 0.0){
            if(gIsU){
              matching_goodADC_stripsU++;
            }
            else if(gIsV){
              matching_goodADC_stripsV++;
            }
          }

          break;
        }
      }

      if(matching_goodADC_stripsU < 1 || matching_goodADC_stripsV < 1){
        nSkippedBadStripMatch++;
        pass_event = false;
      }
    }

    // ----------------------------------------------------
    // Filter 6:
    // Skip event if more than 160 fired regular U or V strips
    // are inside the ROI strip range.
    // ----------------------------------------------------

    if(pass_event){

      int nfired_ustrips_inroi = 0;
      int nfired_vstrips_inroi = 0;

      int ru_min = int(roi_ustrip_min);
      int ru_max = int(roi_ustrip_max);
      int rv_min = int(roi_vstrip_min);
      int rv_max = int(roi_vstrip_max);

      for(int irs = 0; irs < Ns; irs++){

        if(strip_IsU[irs]){
          if(strip_istrip[irs] > ru_min && strip_istrip[irs] < ru_max){
            nfired_ustrips_inroi++;
          }
        }
        else if(strip_IsV[irs]){
          if(strip_istrip[irs] > rv_min && strip_istrip[irs] < rv_max){
            nfired_vstrips_inroi++;
          }
        }
      }

      if(nfired_ustrips_inroi > 160 || nfired_vstrips_inroi > 160){
        nSkippedOver160++;
        pass_event = false;
      }
    }

    // ----------------------------------------------------
    // If event passed all filters, copy the full original event.
    // ----------------------------------------------------

    if(pass_event){
      Tout->Fill();
      nPassed++;
    }
  }

  // ----------------------------------------------------
  // Write output.
  // ----------------------------------------------------

  fout->cd();
  Tout->Write("T", TObject::kOverwrite);
  fout->Close();

  delete T;

  // ----------------------------------------------------
  // Print summary.
  // ----------------------------------------------------

  std::cout << "\nFiltering complete.\n";
  std::cout << "Input events:                              " << nentries << "\n";
  std::cout << "Passed events written to output:           " << nPassed << "\n";
  std::cout << "Skipped: too many hits:                    " << nSkippedTooManyHits << "\n";
  std::cout << "Skipped: roi_inmod == 0:                   " << nSkippedNoROI << "\n";
  std::cout << "Skipped: too many strips:                  " << nSkippedTooManyStrips << "\n";
  std::cout << "Skipped: goodADC strip outside ROI:        " << nSkippedGoodADCStripOutsideROI << "\n";
  std::cout << "Skipped: bad goodADC strip matching:       " << nSkippedBadStripMatch << "\n";
  std::cout << "Skipped: more than 160 strips in ROI:      " << nSkippedOver160 << "\n";
  std::cout << "Output file:                               " << outfile << "\n\n";
}