import com.comsol.model.*;
import com.comsol.model.util.*;

public class EvalStrict3BdfStd1MetricsNoSolve {
  private static double eval(Model m, String dset, String expr, String tag) {
    try {
      try { m.result().numerical().remove(tag); } catch (Exception ignored) {}
      m.result().numerical().create(tag, "MaxVolume");
      m.result().numerical(tag).set("data", dset);
      m.result().numerical(tag).set("expr", new String[]{expr});
      m.result().numerical(tag).selection().all();
      m.result().numerical(tag).setResult();
      double[][] r = m.result().numerical(tag).getReal();
      if (r != null && r.length > 0 && r[0].length > 0) return r[0][0];
    } catch (Exception ignored) {}
    return Double.NaN;
  }

  public static void main(String[] args) throws Exception {
    Model m = ModelUtil.load("Model", "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph");
    double vm = eval(m, "dset6", "solid.mises", "mx_vm_std1_nosolve");
    double um = eval(m, "dset6", "sqrt(u^2+v^2+w^2)", "mx_um_std1_nosolve");
    System.out.println("STD1_NO_SOLVE|vm=" + vm + "|um=" + um + "|finite_nonzero=" + (Double.isFinite(vm) && Double.isFinite(um) && Math.abs(vm) > 1e-12 && Math.abs(um) > 1e-15));
  }
}
