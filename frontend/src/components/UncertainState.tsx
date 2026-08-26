import { HelpCircle, ArrowLeft } from 'lucide-react';
import { Link } from 'react-router-dom';

interface Props {
  message?: string;
}

export default function UncertainState({ message }: Props) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-10 text-center max-w-lg mx-auto my-12">
      <div className="mx-auto w-12 h-12 bg-amber-50 rounded-full flex items-center justify-center mb-5 text-amber-600 border border-amber-100">
        <HelpCircle className="w-6 h-6" />
      </div>

      <h2 className="text-xl font-bold text-slate-900 mb-2">
        Analysis Uncertain
      </h2>

      <p className="text-sm text-slate-600 mb-2 font-medium">
        The model could not classify this image reliably.
      </p>
      <p className="text-xs text-slate-500 mb-8">
        {message || 'Please upload a clearer image showing the leaf surface.'}
      </p>

      <Link
        to="/"
        className="inline-flex justify-center items-center gap-2 px-5 py-2.5 border border-transparent text-sm font-medium rounded-lg text-white bg-emerald-600 hover:bg-emerald-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-emerald-500 transition-colors shadow-sm"
      >
        <ArrowLeft className="w-4 h-4" />
        <span>Upload Another Image</span>
      </Link>
    </div>
  );
}
