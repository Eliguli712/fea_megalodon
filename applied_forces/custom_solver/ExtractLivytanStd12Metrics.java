import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ExtractLivytanStd12Metrics {
  private static final String MPH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_melville_teeth_volsolve.mph";

  private static double evalVolume(Model m, String tag, String type, String expr, String dataset) {
    try { m.result().numerical().remove(tag); } catch (Exception ignored) {}
    try {
      m.result().numerical().create(tag, type);
      m.result().numerical(tag).set("expr", new String[]{expr});
      m.result().numerical(tag).set("data", dataset);
      m.result().numerical(tag).selection().all();
      m.result().numerical(tag).setResult();
      double[][] r = m.result().numerical(tag).getReal();
      if (r != null && r.length > 0 && r[0].length > 0) return r[0][0];
    } catch (Exception ignored) {}
    return Double.NaN;
  }

  public static void main(String[] args) throws Exception {
    Model m;
    try { m = ModelUtil.load("Model", MPH); }
    catch (IOException e) { throw new RuntimeException(e); }

    String ds1 = "dset1";
    String ds2 = "dset2";

    double std1MaxVms = evalVolume(m, "mx_std1_vms_ro", "MaxVolume", "solid.mises", ds1);
    double std1AvgVms = evalVolume(m, "av_std1_vms_ro", "AvVolume", "solid.mises", ds1);
    double std1MaxDisp = evalVolume(m, "mx_std1_u_ro", "MaxVolume", "sqrt(u^2+v^2+w^2)", ds1);
    double std1AvgDisp = evalVolume(m, "av_std1_u_ro", "AvVolume", "sqrt(u^2+v^2+w^2)", ds1);

    double std2MaxVms = evalVolume(m, "mx_std2_vms_ro", "MaxVolume", "solid.mises", ds2);
    double std2AvgVms = evalVolume(m, "av_std2_vms_ro", "AvVolume", "solid.mises", ds2);
    double std2MaxDisp = evalVolume(m, "mx_std2_u_ro", "MaxVolume", "sqrt(u^2+v^2+w^2)", ds2);
    double std2AvgDisp = evalVolume(m, "av_std2_u_ro", "AvVolume", "sqrt(u^2+v^2+w^2)", ds2);

    System.out.println("STD1|max_mises_pa=" + std1MaxVms + "|avg_mises_pa=" + std1AvgVms + "|max_disp_m=" + std1MaxDisp + "|avg_disp_m=" + std1AvgDisp);
    System.out.println("STD2|max_mises_pa=" + std2MaxVms + "|avg_mises_pa=" + std2AvgVms + "|max_disp_m=" + std2MaxDisp + "|avg_disp_m=" + std2AvgDisp);
  }
}
