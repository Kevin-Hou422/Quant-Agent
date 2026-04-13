import React from 'react'

interface Props {
  children: React.ReactNode
  fallback?: React.ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

export default class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('[ErrorBoundary]', error, info.componentStack)
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback
      return (
        <div className="flex flex-col items-center justify-center h-full gap-2 text-rose-400 text-xs px-4 text-center">
          <span className="text-2xl">⚠</span>
          <span className="font-semibold">Chart render error</span>
          <span className="text-slate-500">{this.state.error?.message}</span>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            className="mt-2 px-3 py-1 rounded bg-slate-800 text-slate-300 hover:bg-slate-700 transition-colors"
          >
            Retry
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
