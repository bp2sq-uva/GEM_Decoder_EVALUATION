// hitFindEff_goodADC_skipBadEvents_corrected.C
//
// Usage from shell:
// root -l -b -q 'hitFindEff_goodADC_skipBadEvents_corrected.C++("input.root","eval_hitmid.root","sbs.gemFT.m0")'
//
// Wildcards are OK:
// root -l -b -q 'hitFindEff_goodADC_skipBadEvents_corrected.C++("/path/to/replayed_*.root","eval_hitmid.root","sbs.gemFT.m0")'

#include <TChain.h>
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TString.h>

#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>

bool StripInsideWindow(double strip, double lo, double nstrips){
  double hi = lo + nstrips - 1.0;
  return (lo <= strip && strip <= hi);
}

double ClusterMid(double lo, double nstrips){
  return lo + 0.5*(nstrips - 1.0);
}

void hitFindEff_goodADC_skipBadEvents(
    const char *infile,
    const char *outfile  = "goodADC_2D_benchmark.root",
    const TString prefix = "sbs.gemFT.m0",
    double half_size_x   = 0.75,
    double half_size_y   = 0.20
){
  TChain *T = new TChain("T");
  T->Add(infile);

  if(!T || T->GetEntries() == 0){
    std::cerr << "ERROR: Cannot find replay event tree T, or tree is empty.\n";
    return;
  }

  // Large enough to avoid most overflows. These are heap-allocated below.
  // The event is still skipped before full reading if its counters exceed these limits.
  constexpr int NMAXHITS   = 200000;
  constexpr int NMAXSTRIPS = 500000;

  TString p = prefix;

  // Scalar counters / ROI values.
  double nhit = 0.0;
  double ngood = 0.0;
  double nstrip = 0.0;
  double nstrip_goodADC = 0.0;

  double roi_inmod = 0.0;
  double roi_xmax = 0.0;
  double roi_xmin = 0.0;
  double roi_ymax = 0.0;
  double roi_ymin = 0.0;
  double roi_ustrip_min = 0.0;
  double roi_ustrip_max = 0.0;
  double roi_vstrip_min = 0.0;
  double roi_vstrip_max = 0.0;

  // Heap buffers. Do not put these huge arrays on the stack.
  std::vector<double> hitx(NMAXHITS, 0.0), hity(NMAXHITS, 0.0);
  std::vector<double> hitADC(NMAXHITS, 0.0), hitGoodADC(NMAXHITS, 0.0);
  std::vector<double> goodx(NMAXHITS, 0.0), goody(NMAXHITS, 0.0), goodADC(NMAXHITS, 0.0);
  std::vector<double> hit_iu(NMAXHITS, 0.0), hit_iv(NMAXHITS, 0.0);
  std::vector<double> good_iu(NMAXHITS, 0.0), good_iv(NMAXHITS, 0.0);

  std::vector<double> strip_istrip(NMAXSTRIPS, 0.0), strip_IsU(NMAXSTRIPS, 0.0);
  std::vector<double> strip_IsV(NMAXSTRIPS, 0.0), strip_ADCsum(NMAXSTRIPS, 0.0);

  std::vector<double> good_strip_istrip(NMAXSTRIPS, 0.0), good_strip_IsU(NMAXSTRIPS, 0.0);
  std::vector<double> good_strip_IsV(NMAXSTRIPS, 0.0), good_strip_ADCsum(NMAXSTRIPS, 0.0);

  std::vector<double> clu_lo(NMAXHITS, 0.0), clu_n(NMAXHITS, 0.0), clu_max(NMAXHITS, 0.0);
  std::vector<double> clv_lo(NMAXHITS, 0.0), clv_n(NMAXHITS, 0.0), clv_max(NMAXHITS, 0.0);

  std::vector<double> good_clu_lo(NMAXHITS, 0.0), good_clu_n(NMAXHITS, 0.0), good_clu_max(NMAXHITS, 0.0);
  std::vector<double> good_clv_lo(NMAXHITS, 0.0), good_clv_n(NMAXHITS, 0.0), good_clv_max(NMAXHITS, 0.0);

  // Set branch addresses.
  T->SetBranchAddress(p + ".hit.nhits2d", &nhit);
  T->SetBranchAddress(p + ".hit.hitx", hitx.data());
  T->SetBranchAddress(p + ".hit.hity", hity.data());
  T->SetBranchAddress(p + ".hit.hitADCavg", hitADC.data());
  T->SetBranchAddress(p + ".hit.goodADC_ADCavg", hitGoodADC.data());
  T->SetBranchAddress(p + ".hit.hit_iuclust", hit_iu.data());
  T->SetBranchAddress(p + ".hit.hit_ivclust", hit_iv.data());

  T->SetBranchAddress(p + ".goodADChit.nhits2d", &ngood);
  T->SetBranchAddress(p + ".goodADChit.hitx", goodx.data());
  T->SetBranchAddress(p + ".goodADChit.hity", goody.data());
  T->SetBranchAddress(p + ".goodADChit.goodADC_ADCavg", goodADC.data());
  T->SetBranchAddress(p + ".goodADChit.hit_iuclust", good_iu.data());
  T->SetBranchAddress(p + ".goodADChit.hit_ivclust", good_iv.data());

  T->SetBranchAddress(p + ".strip.nstripsfired", &nstrip);
  T->SetBranchAddress(p + ".strip.istrip", strip_istrip.data());
  T->SetBranchAddress(p + ".strip.IsU", strip_IsU.data());
  T->SetBranchAddress(p + ".strip.IsV", strip_IsV.data());
  T->SetBranchAddress(p + ".strip.ADCsum", strip_ADCsum.data());

  T->SetBranchAddress(p + ".strip.nstripsfired_goodADC", &nstrip_goodADC);
  T->SetBranchAddress(p + ".strip.istrip_goodADC", good_strip_istrip.data());
  T->SetBranchAddress(p + ".strip.IsU_goodADC", good_strip_IsU.data());
  T->SetBranchAddress(p + ".strip.IsV_goodADC", good_strip_IsV.data());
  T->SetBranchAddress(p + ".strip.ADCsum_goodADC", good_strip_ADCsum.data());

  T->SetBranchAddress(p + ".roi.inmod", &roi_inmod);
  T->SetBranchAddress(p + ".roi.xmax", &roi_xmax);
  T->SetBranchAddress(p + ".roi.xmin", &roi_xmin);
  T->SetBranchAddress(p + ".roi.ymax", &roi_ymax);
  T->SetBranchAddress(p + ".roi.ymin", &roi_ymin);
  T->SetBranchAddress(p + ".roi.ustrip_min", &roi_ustrip_min);
  T->SetBranchAddress(p + ".roi.ustrip_max", &roi_ustrip_max);
  T->SetBranchAddress(p + ".roi.vstrip_min", &roi_vstrip_min);
  T->SetBranchAddress(p + ".roi.vstrip_max", &roi_vstrip_max);

  T->SetBranchAddress(p + ".clust.clustu_istriplo", clu_lo.data());
  T->SetBranchAddress(p + ".clust.clustu_strips", clu_n.data());
  T->SetBranchAddress(p + ".clust.clustu_istripmax", clu_max.data());

  T->SetBranchAddress(p + ".clust.clustv_istriplo", clv_lo.data());
  T->SetBranchAddress(p + ".clust.clustv_strips", clv_n.data());
  T->SetBranchAddress(p + ".clust.clustv_istripmax", clv_max.data());

  T->SetBranchAddress(p + ".goodADCclust.clustu_istriplo", good_clu_lo.data());
  T->SetBranchAddress(p + ".goodADCclust.clustu_strips", good_clu_n.data());
  T->SetBranchAddress(p + ".goodADCclust.clustu_istripmax", good_clu_max.data());

  T->SetBranchAddress(p + ".goodADCclust.clustv_istriplo", good_clv_lo.data());
  T->SetBranchAddress(p + ".goodADCclust.clustv_strips", good_clv_n.data());
  T->SetBranchAddress(p + ".goodADCclust.clustv_istripmax", good_clv_max.data());

  // Branch pointers for the safe scalar pre-read.
  TBranch *b_nhit  = T->GetBranch(p + ".hit.nhits2d");
  TBranch *b_ngood = T->GetBranch(p + ".goodADChit.nhits2d");
  TBranch *b_nstrip = T->GetBranch(p + ".strip.nstripsfired");
  TBranch *b_nstrip_goodADC = T->GetBranch(p + ".strip.nstripsfired_goodADC");

  TBranch *b_roi_inmod = T->GetBranch(p + ".roi.inmod");
  TBranch *b_roi_xmax  = T->GetBranch(p + ".roi.xmax");
  TBranch *b_roi_xmin  = T->GetBranch(p + ".roi.xmin");
  TBranch *b_roi_ymax  = T->GetBranch(p + ".roi.ymax");
  TBranch *b_roi_ymin  = T->GetBranch(p + ".roi.ymin");
  TBranch *b_roi_ustrip_min = T->GetBranch(p + ".roi.ustrip_min");
  TBranch *b_roi_ustrip_max = T->GetBranch(p + ".roi.ustrip_max");
  TBranch *b_roi_vstrip_min = T->GetBranch(p + ".roi.vstrip_min");
  TBranch *b_roi_vstrip_max = T->GetBranch(p + ".roi.vstrip_max");

  if(!b_nhit || !b_ngood || !b_nstrip || !b_nstrip_goodADC ||
     !b_roi_inmod || !b_roi_xmax || !b_roi_xmin || !b_roi_ymax || !b_roi_ymin ||
     !b_roi_ustrip_min || !b_roi_ustrip_max || !b_roi_vstrip_min || !b_roi_vstrip_max){
    std::cerr << "ERROR: One or more required scalar/count/ROI branches are missing.\n";
    std::cerr << "Check prefix: " << p << "\n";
    return;
  }

  TH1D *h_best_score = new TH1D(
      "h_best_score",
      "best accepted hit distance in strip-midpoint space;#DeltaU^{2}+#DeltaV^{2};truth hits",
      200, 0, 100
  );

  TH1D *h_eff_num = new TH1D(
      "h_eff_num",
      "matched good-ADC hits;x_{good} [m];matched",
      120, -half_size_x, half_size_x
  );

  TH1D *h_eff_den = new TH1D(
      "h_eff_den",
      "all good-ADC hits;x_{good} [m];truth",
      120, -half_size_x, half_size_x
  );

  TH2D *h_truth_xy = new TH2D(
      "h_truth_xy",
      "good-ADC truth hits;x [m];y [m]",
      120, -half_size_x, half_size_x,
      120, -half_size_y, half_size_y
  );

  TH2D *h_match_xy = new TH2D(
      "h_match_xy",
      "matched good-ADC truth hits;x [m];y [m]",
      120, -half_size_x, half_size_x,
      120, -half_size_y, half_size_y
  );

  TH1D *h_purity_best = new TH1D(
      "h_purity_best",
      "good-ADC fraction of selected regular hit;goodADC_{hit}/ADC_{hit};matches",
      100, 0, 1.2
  );

  TH1D *h_weight = new TH1D(
      "h_weight",
      "purity-weighted efficiency contribution;w;truth hits",
      100, 0, 1.2
  );

  Long64_t nentries = T->GetEntries();

  Long64_t nTruth = 0;
  Long64_t nMatched = 0;
  Long64_t nSkippedTooManyHits = 0;
  Long64_t nSkippedTooManyStrips = 0;
  Long64_t nSkippedBadStrip = 0;
  Long64_t nSkippedOver160 = 0;
  double sumWeight = 0.0;

  for(Long64_t ev = 0; ev < nentries; ev++){

    // IMPORTANT: read only small scalar branches first.
    // Do not call T->GetEntry(ev) until after these size checks pass.
    b_nhit->GetEntry(ev);
    b_ngood->GetEntry(ev);
    b_nstrip->GetEntry(ev);
    b_nstrip_goodADC->GetEntry(ev);

    b_roi_inmod->GetEntry(ev);
    b_roi_xmax->GetEntry(ev);
    b_roi_xmin->GetEntry(ev);
    b_roi_ymax->GetEntry(ev);
    b_roi_ymin->GetEntry(ev);
    b_roi_ustrip_min->GetEntry(ev);
    b_roi_ustrip_max->GetEntry(ev);
    b_roi_vstrip_min->GetEntry(ev);
    b_roi_vstrip_max->GetEntry(ev);

    int Nh  = int(nhit);
    int Ng  = int(ngood);
    int Ns  = int(nstrip);
    int Nsg = int(nstrip_goodADC);

    if(Nh < 0 || Ng < 0 || Ns < 0 || Nsg < 0){
      std::cerr << "Event " << ev << " has negative branch counts. Skipping event.\n";
      continue;
    }

    if(Nh > NMAXHITS || Ng > NMAXHITS){
      nSkippedTooManyHits++;
      std::cerr << "Event " << ev << " has too many hits: "
                << "Nh = " << Nh << ", Ng = " << Ng
                << ", NMAXHITS = " << NMAXHITS
                << ". Skipping event before reading hit arrays.\n";
      continue;
    }

    if(Ns > NMAXSTRIPS || Nsg > NMAXSTRIPS){
      nSkippedTooManyStrips++;
      std::cerr << "Event " << ev << " has too many strips: "
                << "Ns = " << Ns << ", Nsg = " << Nsg
                << ", NMAXSTRIPS = " << NMAXSTRIPS
                << ". Skipping event before reading strip arrays.\n";
      continue;
    }

    if(roi_inmod == 0.0) continue;

    // Only now is it safe to read the full event, including large array branches.
    T->GetEntry(ev);

    // Refresh integer counters after full read.
    Nh  = int(nhit);
    Ng  = int(ngood);
    Ns  = int(nstrip);
    Nsg = int(nstrip_goodADC);

    bool bad_event = false;

    // Skip events where positive good-ADC strips do not have at least one matching
    // regular positive strip in both U and V.
    int matching_goodADC_stripsU = 0;
    int matching_goodADC_stripsV = 0;

    for(int igs = 0; igs < Nsg; igs++){
      if(good_strip_ADCsum[igs] <= 0.0) continue;

      int gstrip = int(good_strip_istrip[igs]);
      int gIsU   = int(good_strip_IsU[igs]);
      int gIsV   = int(good_strip_IsV[igs]);

      for(int irs = 0; irs < Ns; irs++){
        bool same_axis =
          (int(strip_IsU[irs]) == gIsU) &&
          (int(strip_IsV[irs]) == gIsV);

        bool same_strip = (int(strip_istrip[irs]) == gstrip);

        if(!same_axis || !same_strip) continue;

        if(strip_ADCsum[irs] > 0.0){
          if(gIsU) matching_goodADC_stripsU++;
          else     matching_goodADC_stripsV++;
        }

        break;
      }
    }

    if(matching_goodADC_stripsU < 1 || matching_goodADC_stripsV < 1){
      nSkippedBadStrip++;
      continue;
    }

    // Skip high-occupancy events inside ROI.
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
      } else if(strip_IsV[irs]){
        if(strip_istrip[irs] > rv_min && strip_istrip[irs] < rv_max){
          nfired_vstrips_inroi++;
        }
      }
    }

    if(nfired_ustrips_inroi > 160 || nfired_vstrips_inroi > 160){
      nSkippedOver160++;
      continue;
    }

    // Loop over good-ADC truth hits.
    for(int ig = 0; ig < Ng; ig++){
      double xg = goodx[ig];
      double yg = goody[ig];

      // This good-ADC hit is outside the ROI, so do not count it in the denominator.
      if(xg < roi_xmin || xg > roi_xmax || yg < roi_ymin || yg > roi_ymax) continue;

      h_eff_den->Fill(xg);
      h_truth_xy->Fill(xg, yg);
      nTruth++;

      int best = -1;
      double bestScore = std::numeric_limits<double>::max();

      int goodUidx = int(good_iu[ig]);
      int goodVidx = int(good_iv[ig]);

      if(goodUidx < 0 || goodVidx < 0 ||
         goodUidx >= NMAXHITS || goodVidx >= NMAXHITS){
        h_weight->Fill(0.0);
        std::cerr << "Event " << ev << ", good-ADC hit " << ig
                  << " has invalid cluster indices: "
                  << "goodUidx = " << goodUidx << ", goodVidx = " << goodVidx
                  << ". Skipping this hit.\n";
        continue;
      }

      double goodUlo = good_clu_lo[goodUidx];
      double goodUn  = good_clu_n[goodUidx];
      double goodVlo = good_clv_lo[goodVidx];
      double goodVn  = good_clv_n[goodVidx];

      double goodUmid = ClusterMid(goodUlo, goodUn);
      double goodVmid = ClusterMid(goodVlo, goodVn);

      for(int ih = 0; ih < Nh; ih++){
        int recoUidx = int(hit_iu[ih]);
        int recoVidx = int(hit_iv[ih]);

        if(recoUidx < 0 || recoVidx < 0 ||
           recoUidx >= NMAXHITS || recoVidx >= NMAXHITS){
          continue;
        }

        double recoUmax = clu_max[recoUidx];
        double recoVmax = clv_max[recoVidx];
        
        double recoUmid = ClusterMid(clu_lo[recoUidx], clu_n[recoUidx]);
        double recoVmid = ClusterMid(clv_lo[recoVidx], clv_n[recoVidx]);

        bool passU = StripInsideWindow(recoUmax, goodUlo, goodUn);
        bool passV = StripInsideWindow(recoVmax, goodVlo, goodVn);

        if(!passU || !passV) continue;

        double dU = recoUmid - goodUmid;
        double dV = recoVmid - goodVmid;
        double score = dU*dU + dV*dV;

        if(score < bestScore){
          bestScore = score;
          best = ih;
        }
      }

      if(best >= 0){
        nMatched++;

        h_best_score->Fill(bestScore);
        h_eff_num->Fill(xg);
        h_match_xy->Fill(xg, yg);

        double purity = 0.0;
        if(hitADC[best] > 0.0){
          purity = hitGoodADC[best] / hitADC[best];
        }

        purity = std::max(0.0, std::min(1.0, purity));

        h_purity_best->Fill(purity);
        h_weight->Fill(purity);
        sumWeight += purity;
      } else {
        h_weight->Fill(0.0);
      }
    }
  }

  double eff = (nTruth > 0) ? double(nMatched)/double(nTruth) : 0.0;
  double weightedEff = (nTruth > 0) ? sumWeight/double(nTruth) : 0.0;

  std::cout << "Truth good-ADC 2D hits: " << nTruth << "\n";
  std::cout << "Matched regular 2D hits: " << nMatched << "\n";
  std::cout << "ML-style strip-window efficiency: " << eff << "\n";
  std::cout << "Good-ADC-purity-weighted efficiency: " << weightedEff << "\n";
  std::cout << "Events skipped due to too many hits: " << nSkippedTooManyHits << "\n";
  std::cout << "Events skipped due to too many strips: " << nSkippedTooManyStrips << "\n";
  std::cout << "Events skipped due to missing matching regular U/V good-ADC strips: "
            << nSkippedBadStrip << "\n";
  std::cout << "Events skipped due to fired strips in ROI exceeding 160 in U or V: "
            << nSkippedOver160 << "\n";

  TFile *fout = TFile::Open(outfile, "RECREATE");
  if(!fout || fout->IsZombie()){
    std::cerr << "ERROR: Could not open output file " << outfile << "\n";
    return;
  }

  h_best_score->Write();
  h_eff_num->Write();
  h_eff_den->Write();
  h_truth_xy->Write();
  h_match_xy->Write();
  h_purity_best->Write();
  h_weight->Write();

  TH1D *h_eff_x = (TH1D*)h_eff_num->Clone("h_eff_x");
  h_eff_x->SetTitle("ML-style 2D hit-finding efficiency vs x;x_{good} [m];efficiency");
  h_eff_x->Divide(h_eff_den);
  h_eff_x->Write();

  fout->Close();
}
